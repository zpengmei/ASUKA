"""Tests for orbital tracking across geometry changes."""

from __future__ import annotations

import numpy as np
import pytest


def test_cross_geometry_overlap_identity():
    """Cross-geometry overlap should equal regular overlap for identical geometries."""
    from asuka.frontend import Molecule
    from asuka.frontend.one_electron import build_ao_basis_cart
    from asuka.integrals.cross_geometry import build_S_cross_cart
    from asuka.integrals.int1e_cart import build_S_cart

    mol = Molecule.from_atoms(
        atoms=[("H", (0, 0, 0)), ("H", (0, 0, 1.4))], basis="sto-3g"
    )
    basis, _ = build_ao_basis_cart(mol)

    S = build_S_cart(basis)
    S_cross = build_S_cross_cart(basis, basis)

    assert S.shape == S_cross.shape
    assert np.allclose(S, S_cross, atol=1e-12)


def test_cross_geometry_overlap_different_geometries():
    """Cross-geometry overlap should differ for different geometries."""
    from asuka.frontend import Molecule
    from asuka.frontend.one_electron import build_ao_basis_cart
    from asuka.integrals.cross_geometry import build_S_cross_cart

    mol1 = Molecule.from_atoms(
        atoms=[("H", (0, 0, 0)), ("H", (0, 0, 1.4))], basis="sto-3g"
    )
    mol2 = Molecule.from_atoms(
        atoms=[("H", (0, 0, 0)), ("H", (0, 0, 2.0))], basis="sto-3g"
    )

    basis1, _ = build_ao_basis_cart(mol1)
    basis2, _ = build_ao_basis_cart(mol2)

    S_cross = build_S_cross_cart(basis1, basis2)

    # Overlap should be close to identity but not exact
    assert S_cross.shape == (2, 2)  # sto-3g H2 has 2 AOs
    assert not np.allclose(S_cross, np.eye(2), atol=1e-3)
    # But diagonal elements should still be large (similar orbitals)
    assert np.all(np.diag(S_cross) > 0.9)


def test_mo_overlap_orthonormal():
    """MO overlap with self using identity AO overlap should be identity."""
    from asuka.mcscf.orbital_tracking import compute_mo_overlap

    nao, nmo = 10, 10
    C = np.eye(nmo, dtype=np.float64)
    S = np.eye(nao, dtype=np.float64)

    O = compute_mo_overlap(C, C, S)

    assert O.shape == (nmo, nmo)
    assert np.allclose(O, np.eye(nmo), atol=1e-12)


def test_mo_overlap_rectangular():
    """MO overlap should handle different number of MOs."""
    from asuka.mcscf.orbital_tracking import compute_mo_overlap

    nao = 10
    nmo_prev = 8
    nmo_new = 10
    C_prev = np.random.rand(nao, nmo_prev)
    C_new = np.random.rand(nao, nmo_new)
    S = np.eye(nao)

    O = compute_mo_overlap(C_prev, C_new, S)

    assert O.shape == (nmo_prev, nmo_new)


def test_assign_active_orbitals_subspace_simple():
    """Subspace method should pick orbitals with max overlap to active space."""
    from asuka.mcscf.orbital_tracking import assign_active_orbitals_by_overlap

    # Create mock case where prev active=[2,3,4] has high overlap with new [3,4,5]
    nmo = 10
    C_prev = np.eye(nmo)
    C_new = np.eye(nmo)
    # Swap orbitals: 2->3, 3->4, 4->5 (shift active space by 1)
    C_new[:, [2, 3, 4, 5]] = C_new[:, [5, 2, 3, 4]]
    S = np.eye(nmo)

    prev_active = [2, 3, 4]
    new_active = assign_active_orbitals_by_overlap(
        C_prev, C_new, S, prev_active, ncas=3, method="subspace"
    )

    assert len(new_active) == 3
    assert set(new_active) == {3, 4, 5}


def test_assign_active_orbitals_subspace_no_shift():
    """Subspace method should pick same orbitals if no shift."""
    from asuka.mcscf.orbital_tracking import assign_active_orbitals_by_overlap

    nmo = 10
    C_prev = np.eye(nmo)
    C_new = np.eye(nmo)
    S = np.eye(nmo)

    prev_active = [3, 4, 5]
    new_active = assign_active_orbitals_by_overlap(
        C_prev, C_new, S, prev_active, ncas=3, method="subspace"
    )

    assert len(new_active) == 3
    assert set(new_active) == {3, 4, 5}


def test_assign_active_orbitals_hungarian():
    """Hungarian method should give optimal 1-to-1 assignment."""
    pytest.importorskip("scipy")  # Skip if scipy not available

    from asuka.mcscf.orbital_tracking import assign_active_orbitals_by_overlap

    nmo = 10
    C_prev = np.eye(nmo)
    C_new = np.eye(nmo)
    # Swap orbitals 2<->5
    C_new[:, [2, 5]] = C_new[:, [5, 2]]
    S = np.eye(nmo)

    prev_active = [2, 3, 4]
    new_active = assign_active_orbitals_by_overlap(
        C_prev, C_new, S, prev_active, ncas=3, method="hungarian"
    )

    assert len(new_active) == 3
    # Should find that prev [2,3,4] maps to new [5,3,4]
    assert set(new_active) == {3, 4, 5}


def test_align_orbital_phases():
    """Phase alignment should give positive diagonal overlaps."""
    from asuka.mcscf.orbital_tracking import align_orbital_phases, compute_mo_overlap

    nmo = 5
    C_prev = np.eye(nmo)
    C_new = np.eye(nmo)
    C_new[:, 2] *= -1  # Flip one orbital
    C_new[:, 4] *= -1  # Flip another
    S = np.eye(nmo)

    # Before alignment, some diagonal overlaps are negative
    O_before = compute_mo_overlap(C_prev, C_new, S)
    assert O_before[2, 2] < 0
    assert O_before[4, 4] < 0

    # After alignment, all diagonal overlaps should be positive
    C_aligned = align_orbital_phases(C_prev, C_new, S)
    O_after = compute_mo_overlap(C_prev, C_aligned, S)

    assert np.all(np.diag(O_after) > 0)
    assert np.allclose(np.abs(np.diag(O_after)), 1.0, atol=1e-12)


def test_align_orbital_phases_partial():
    """Phase alignment with specific indices should only align those orbitals."""
    from asuka.mcscf.orbital_tracking import align_orbital_phases, compute_mo_overlap

    nmo = 5
    C_prev = np.eye(nmo)
    C_new = np.eye(nmo)
    C_new[:, 1] *= -1
    C_new[:, 2] *= -1
    C_new[:, 3] *= -1
    S = np.eye(nmo)

    # Only align orbitals 2 and 3
    C_aligned = align_orbital_phases(C_prev, C_new, S, alignment_idx=[2, 3])
    O = compute_mo_overlap(C_prev, C_aligned, S)

    # Orbitals 2 and 3 should be positive
    assert O[2, 2] > 0
    assert O[3, 3] > 0
    # Orbital 1 should still be negative (not aligned)
    assert O[1, 1] < 0


def test_reorder_mo_to_active_space():
    """Reordering should place specified orbitals in active space."""
    from asuka.mcscf.orbital_tracking import reorder_mo_to_active_space

    nmo = 10
    mo = np.eye(nmo)

    # Want orbitals [5,6,7] to become the active space at positions [2,3,4]
    mo_reordered = reorder_mo_to_active_space(mo, active_idx=[5, 6, 7], ncore=2)

    # Check that positions 2,3,4 now contain original orbitals 5,6,7
    assert np.allclose(mo_reordered[:, 2], mo[:, 5])
    assert np.allclose(mo_reordered[:, 3], mo[:, 6])
    assert np.allclose(mo_reordered[:, 4], mo[:, 7])


def test_reorder_mo_to_active_space_preserves_orthonormality():
    """Reordering should preserve orthonormality of MOs."""
    from asuka.mcscf.orbital_tracking import reorder_mo_to_active_space

    nmo = 10
    # Random orthonormal matrix
    mo = np.linalg.qr(np.random.rand(nmo, nmo))[0]

    mo_reordered = reorder_mo_to_active_space(mo, active_idx=[3, 5, 7], ncore=1)

    # Should still be orthonormal
    assert np.allclose(mo_reordered.T @ mo_reordered, np.eye(nmo), atol=1e-12)


def test_h2_dissociation_tracking():
    """Orbital tracking should prevent active space drift in H2 scan."""
    pytest.importorskip("cupy")  # Skip if CuPy not available (DF requires it)

    from asuka.frontend import Molecule, make_df_casscf_energy_grad

    mol = Molecule.from_atoms(
        atoms=[("H", (0, 0, 0)), ("H", (0, 0, 1.4))], basis="sto-3g"
    )

    # Create energy_grad function with orbital tracking enabled
    energy_grad = make_df_casscf_energy_grad(
        mol,
        hf_kwargs={"backend": "cpu"},
        casscf_kwargs={
            "ncore": 0,
            "ncas": 2,
            "nelecas": 2,
            "max_cycle_macro": 20,
            "backend": "cpu",
        },
        grad_kwargs={"df_backend": "cpu"},
        orbital_tracking=True,
        tracking_method="subspace",
    )

    # Scan H-H distance
    coords_list = [
        np.array([[0, 0, 0], [0, 0, 1.4]]),
        np.array([[0, 0, 0], [0, 0, 2.0]]),
        np.array([[0, 0, 0], [0, 0, 3.0]]),
    ]

    energies = []
    for coords in coords_list:
        E, grad = energy_grad(coords.flatten())
        energies.append(E)

    # Energies should be monotonically increasing (dissociation)
    assert energies[1] > energies[0]
    assert energies[2] > energies[1]

    # Energy changes should be smooth (no sudden jumps from orbital reordering)
    # For H2 dissociation, typical energy change is ~0.1 Eh per step
    assert abs(energies[1] - energies[0]) < 0.2
    assert abs(energies[2] - energies[1]) < 0.2


def test_orbital_tracking_disabled():
    """When orbital tracking is disabled, it should behave as before."""
    pytest.importorskip("cupy")  # Skip if CuPy not available (DF requires it)

    from asuka.frontend import Molecule, make_df_casscf_energy_grad

    mol = Molecule.from_atoms(
        atoms=[("H", (0, 0, 0)), ("H", (0, 0, 1.4))], basis="sto-3g"
    )

    # Create energy_grad function with orbital tracking disabled
    energy_grad = make_df_casscf_energy_grad(
        mol,
        hf_kwargs={"backend": "cpu"},
        casscf_kwargs={
            "ncore": 0,
            "ncas": 2,
            "nelecas": 2,
            "max_cycle_macro": 20,
            "backend": "cpu",
        },
        grad_kwargs={"df_backend": "cpu"},
        orbital_tracking=False,
    )

    # Should still work
    coords = np.array([[0, 0, 0], [0, 0, 1.4]])
    E, grad = energy_grad(coords.flatten())

    assert isinstance(E, float)
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (6,)


def test_orbital_tracking_requires_ncore_ncas():
    """Orbital tracking should raise error if ncore/ncas not provided."""
    from asuka.frontend import Molecule, make_df_casscf_energy_grad

    mol = Molecule.from_atoms(
        atoms=[("H", (0, 0, 0)), ("H", (0, 0, 1.4))], basis="sto-3g"
    )

    # Should raise ValueError if ncore/ncas not in casscf_kwargs
    with pytest.raises(ValueError, match="orbital_tracking=True requires"):
        make_df_casscf_energy_grad(
            mol,
            casscf_kwargs={"nelecas": 2},  # Missing ncore and ncas
            orbital_tracking=True,
        )


def test_orbital_tracking_with_explicit_ref_mol():
    """Test tracking_ref parameter with explicit Molecule reference."""
    from asuka.frontend import Molecule, make_df_casscf_energy_grad

    # Reference molecule at equilibrium
    ref_mol = Molecule.from_atoms(
        atoms=[("H", (0, 0, 0)), ("H", (0, 0, 1.4))], basis="sto-3g"
    )

    # New molecule at different geometry
    mol = Molecule.from_atoms(
        atoms=[("H", (0, 0, 0)), ("H", (0, 0, 2.0))], basis="sto-3g"
    )

    # Should accept tracking_ref=Molecule
    energy_grad = make_df_casscf_energy_grad(
        mol,
        casscf_kwargs={"ncore": 0, "ncas": 2, "nelecas": 2},
        orbital_tracking=True,
        tracking_ref=ref_mol,
    )

    # Should create successfully
    assert callable(energy_grad)


def test_orbital_tracking_with_tuple_ref():
    """Test tracking_ref parameter with (mol, result) tuple."""
    from asuka.frontend import Molecule, make_df_casscf_energy_grad
    from dataclasses import dataclass

    # Mock CASSCF result
    @dataclass
    class MockCASSCF:
        mol: Molecule
        mo_coeff: np.ndarray
        ci: np.ndarray
        ncore: int
        ncas: int

    ref_mol = Molecule.from_atoms(
        atoms=[("H", (0, 0, 0)), ("H", (0, 0, 1.4))], basis="sto-3g"
    )
    ref_result = MockCASSCF(
        mol=ref_mol,
        mo_coeff=np.eye(2),
        ci=np.array([1.0, 0.0]),
        ncore=0,
        ncas=2,
    )

    mol = Molecule.from_atoms(
        atoms=[("H", (0, 0, 0)), ("H", (0, 0, 2.0))], basis="sto-3g"
    )

    # Should accept tracking_ref=(mol, result)
    energy_grad = make_df_casscf_energy_grad(
        mol,
        casscf_kwargs={"ncore": 0, "ncas": 2, "nelecas": 2},
        orbital_tracking=True,
        tracking_ref=(ref_mol, ref_result),
    )

    assert callable(energy_grad)


def test_cross_geometry_overlap_water():
    """Test cross-geometry overlap on a realistic molecule."""
    from asuka.frontend import Molecule
    from asuka.frontend.one_electron import build_ao_basis_cart
    from asuka.integrals.cross_geometry import build_S_cross_cart

    # Water at two different OH bond lengths
    mol1 = Molecule.from_atoms(
        atoms=[
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.757, 0.586)),
            ("H", (0.0, -0.757, 0.586)),
        ],
        basis="sto-3g",
    )

    mol2 = Molecule.from_atoms(
        atoms=[
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.8, 0.6)),  # Slightly different
            ("H", (0.0, -0.8, 0.6)),
        ],
        basis="sto-3g",
    )

    basis1, _ = build_ao_basis_cart(mol1)
    basis2, _ = build_ao_basis_cart(mol2)

    S_cross = build_S_cross_cart(basis1, basis2)

    # Should be close to identity (small geometry change)
    nao = S_cross.shape[0]
    assert S_cross.shape == (nao, nao)
    # Diagonal elements should be large
    assert np.all(np.abs(np.diag(S_cross)) > 0.95)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
