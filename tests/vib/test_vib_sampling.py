"""Tests for Wigner/classical normal-mode sampling with momentum and XYZ I/O."""

from __future__ import annotations

import numpy as np
import pytest

from asuka.vib.constants import AMU_TO_AU, AU_TIME_TO_FS, BOHR_TO_ANGSTROM, KB_HARTREE_PER_K
from asuka.vib.frequency import NormalModes, frequency_analysis
from asuka.vib.sampling import WignerSample, sample_normal_modes


# ---------------------------------------------------------------------------
# Helpers: build a simple diatomic (H2) normal-mode object
# ---------------------------------------------------------------------------

def _make_h2_modes() -> NormalModes:
    """H2 with a synthetic Hessian giving a known stretching frequency."""
    # Two H atoms along z-axis, 1.4 Bohr apart
    coords = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    masses = np.array([1.00794, 1.00794])

    # Build a Hessian with a known force constant k along z
    # omega = sqrt(k / mu_au), mu = m/2 for homonuclear diatomic
    k = 0.5  # Hartree/Bohr^2
    n = 6
    H = np.zeros((n, n), dtype=np.float64)
    # d^2E/dz1^2 = k, d^2E/dz2^2 = k, d^2E/dz1dz2 = -k
    H[2, 2] = k
    H[5, 5] = k
    H[2, 5] = -k
    H[5, 2] = -k

    return frequency_analysis(hessian_cart=H, coords_bohr=coords, masses_amu=masses)


def _make_water_modes() -> NormalModes:
    """Water-like molecule with a synthetic diagonal Hessian."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.43, 1.1],
        [0.0, -1.43, 1.1],
    ])
    masses = np.array([15.999, 1.00794, 1.00794])

    n = 9
    H = np.zeros((n, n), dtype=np.float64)
    # Simple diagonal force constants
    for i in range(n):
        H[i, i] = 0.3 + 0.1 * i

    return frequency_analysis(hessian_cart=H, coords_bohr=coords, masses_amu=masses)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_returns_ndarray_by_default(self):
        modes = _make_h2_modes()
        result = sample_normal_modes(modes, n_samples=5, seed=42)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 2, 3)

    def test_same_seed_same_coords(self):
        modes = _make_h2_modes()
        r1 = sample_normal_modes(modes, n_samples=10, seed=123)
        r2 = sample_normal_modes(modes, n_samples=10, seed=123)
        np.testing.assert_array_equal(r1, r2)

    def test_positions_unchanged_with_return_velocities_false(self):
        """Ensure return_velocities=False gives identical results to the old API."""
        modes = _make_h2_modes()
        r_old = sample_normal_modes(modes, n_samples=10, seed=99)
        r_new = sample_normal_modes(modes, n_samples=10, seed=99, return_velocities=False)
        np.testing.assert_array_equal(r_old, r_new)


# ---------------------------------------------------------------------------
# Momentum / velocity sampling
# ---------------------------------------------------------------------------

class TestWignerMomentum:
    def test_returns_wigner_sample(self):
        modes = _make_h2_modes()
        result = sample_normal_modes(modes, n_samples=5, seed=42, return_velocities=True)
        assert isinstance(result, WignerSample)
        assert result.coords.shape == (5, 2, 3)
        assert result.velocities.shape == (5, 2, 3)

    def test_wigner_momentum_variance_t0(self):
        """At T=0, Var(P_k) should equal omega_k / 2 for each mode."""
        modes = _make_h2_modes()
        n_samples = 200_000

        result = sample_normal_modes(
            modes,
            n_samples=n_samples,
            temperature_k=0.0,
            method="wigner",
            seed=0,
            return_velocities=True,
        )

        # Extract mass-weighted momenta in normal-mode space
        m_au = modes.masses_amu * AMU_TO_AU
        sqrt_m = np.sqrt(np.repeat(m_au, 3))

        vel_flat = result.velocities.reshape((n_samples, -1))
        # v_i = P_mw_i / sqrt(m_i), so P_mw_i = v_i * sqrt(m_i)
        p_mw_flat = vel_flat * sqrt_m[None, :]

        # Project onto vibrational eigenvectors
        lam = modes.eigvals
        keep = lam > 0.0
        Vmw_k = modes.eigvecs_mw[:, keep]
        omega = np.sqrt(lam[keep])

        P_k = p_mw_flat @ Vmw_k  # (n_samples, n_vib)
        var_P = np.var(P_k, axis=0)
        expected = omega / 2.0

        np.testing.assert_allclose(var_P, expected, rtol=0.05)

    def test_uncertainty_product_t0(self):
        """At T=0, Var(Q_k)*Var(P_k) should be 1/4 (minimum uncertainty)."""
        modes = _make_h2_modes()
        n_samples = 200_000

        result = sample_normal_modes(
            modes,
            n_samples=n_samples,
            temperature_k=0.0,
            method="wigner",
            seed=1,
            return_velocities=True,
        )

        m_au = modes.masses_amu * AMU_TO_AU
        sqrt_m = np.sqrt(np.repeat(m_au, 3))

        coords_flat = result.coords.reshape((n_samples, -1))
        eq_flat = modes.coords_bohr.reshape((-1,))
        dR_flat = coords_flat - eq_flat[None, :]
        q_mw_flat = dR_flat * sqrt_m[None, :]

        vel_flat = result.velocities.reshape((n_samples, -1))
        p_mw_flat = vel_flat * sqrt_m[None, :]

        lam = modes.eigvals
        keep = lam > 0.0
        Vmw_k = modes.eigvecs_mw[:, keep]

        Q_k = q_mw_flat @ Vmw_k
        P_k = p_mw_flat @ Vmw_k

        product = np.var(Q_k, axis=0) * np.var(P_k, axis=0)
        np.testing.assert_allclose(product, 0.25, rtol=0.05)

    def test_classical_momentum_variance(self):
        """Classical: Var(P_k) = kB*T, independent of omega."""
        modes = _make_water_modes()
        T = 300.0
        n_samples = 200_000

        result = sample_normal_modes(
            modes,
            n_samples=n_samples,
            temperature_k=T,
            method="classical",
            seed=2,
            return_velocities=True,
        )

        m_au = modes.masses_amu * AMU_TO_AU
        sqrt_m = np.sqrt(np.repeat(m_au, 3))

        vel_flat = result.velocities.reshape((n_samples, -1))
        p_mw_flat = vel_flat * sqrt_m[None, :]

        lam = modes.eigvals
        keep = lam > 0.0
        Vmw_k = modes.eigvecs_mw[:, keep]

        P_k = p_mw_flat @ Vmw_k
        var_P = np.var(P_k, axis=0)
        expected = KB_HARTREE_PER_K * T

        np.testing.assert_allclose(var_P, expected, rtol=0.05)


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

class TestUnitConversion:
    def test_angstrom_coords(self):
        modes = _make_h2_modes()
        r_bohr = sample_normal_modes(modes, n_samples=3, seed=10, unit="Bohr")
        r_ang = sample_normal_modes(modes, n_samples=3, seed=10, unit="Angstrom")
        np.testing.assert_allclose(r_ang, r_bohr * BOHR_TO_ANGSTROM)

    def test_angstrom_velocities(self):
        modes = _make_h2_modes()
        s_bohr = sample_normal_modes(
            modes, n_samples=3, seed=10, unit="Bohr", return_velocities=True,
        )
        s_ang = sample_normal_modes(
            modes, n_samples=3, seed=10, unit="Angstrom", return_velocities=True,
        )
        np.testing.assert_allclose(
            s_ang.velocities,
            s_bohr.velocities * (BOHR_TO_ANGSTROM / AU_TIME_TO_FS),
        )
        np.testing.assert_allclose(s_ang.coords, s_bohr.coords * BOHR_TO_ANGSTROM)


# ---------------------------------------------------------------------------
# NormalModes.sample() convenience method
# ---------------------------------------------------------------------------

class TestNormalModesSample:
    def test_sample_delegates(self):
        modes = _make_h2_modes()
        r1 = sample_normal_modes(modes, n_samples=5, seed=42)
        r2 = modes.sample(n_samples=5, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_sample_with_velocities(self):
        modes = _make_h2_modes()
        result = modes.sample(n_samples=5, seed=42, return_velocities=True)
        assert isinstance(result, WignerSample)


# ---------------------------------------------------------------------------
# XYZ I/O
# ---------------------------------------------------------------------------

class TestXYZIO:
    def test_write_ensemble_xyz_coords_only(self, tmp_path):
        from asuka.vib.io import write_ensemble_xyz

        symbols = ["H", "H"]
        coords = np.array([
            [[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]],
            [[0.0, 0.0, -0.38], [0.0, 0.0, 0.38]],
        ])

        out = tmp_path / "test.xyz"
        write_ensemble_xyz(out, symbols, coords)

        lines = out.read_text().strip().split("\n")
        # Frame 0: lines 0(natm), 1(comment), 2(atom0), 3(atom1)
        # Frame 1: lines 4(natm), 5(comment), 6(atom0), 7(atom1)
        assert lines[0].strip() == "2"
        assert "sample=0" in lines[1]
        assert lines[4].strip() == "2"
        assert "sample=1" in lines[5]

        # Parse back first frame second atom z-coordinate
        tok = lines[3].split()
        assert tok[0] == "H"
        np.testing.assert_allclose(float(tok[3]), 0.37, atol=1e-8)

    def test_write_ensemble_xyz_with_velocities(self, tmp_path):
        from asuka.vib.io import write_ensemble_xyz

        symbols = ["O", "H", "H"]
        coords = np.random.default_rng(0).standard_normal((2, 3, 3))
        vels = np.random.default_rng(1).standard_normal((2, 3, 3))

        out = tmp_path / "test_vel.xyz"
        write_ensemble_xyz(out, symbols, coords, velocities=vels)

        lines = out.read_text().strip().split("\n")
        # Each frame: 1 header + 1 comment + 3 atoms = 5 lines, 2 frames = 10
        assert len(lines) == 10

        # Check that velocity columns are present
        tok = lines[2].split()
        assert len(tok) == 7  # sym + 3 coords + 3 vels

    def test_write_ensemble_xyz_with_energies(self, tmp_path):
        from asuka.vib.io import write_ensemble_xyz

        symbols = ["H", "H"]
        coords = np.zeros((1, 2, 3))

        out = tmp_path / "test_e.xyz"
        write_ensemble_xyz(out, symbols, coords, energies=[-1.123456])

        lines = out.read_text().strip().split("\n")
        assert "E=-1.1234560000" in lines[1]

    def test_write_wigner_ensemble_xyz(self, tmp_path):
        from asuka.vib.io import write_wigner_ensemble_xyz

        # Minimal mock molecule
        class MockMol:
            atoms_bohr = (("H", np.zeros(3)), ("H", np.array([0.0, 0.0, 1.4])))

        coords_bohr = np.zeros((2, 2, 3)) + 0.5
        vels_bohr_au = np.ones((2, 2, 3)) * 0.01
        sample = WignerSample(coords=coords_bohr, velocities=vels_bohr_au)

        out = tmp_path / "wigner.xyz"
        write_wigner_ensemble_xyz(out, MockMol(), sample)

        lines = out.read_text().strip().split("\n")
        assert lines[0].strip() == "2"
        # Check unit conversion happened (coords ~ 0.5 * 0.529 ≈ 0.265 Angstrom)
        tok = lines[2].split()
        x_val = float(tok[1])
        np.testing.assert_allclose(x_val, 0.5 * BOHR_TO_ANGSTROM, atol=1e-6)

    def test_write_wigner_ensemble_xyz_bare_array(self, tmp_path):
        from asuka.vib.io import write_wigner_ensemble_xyz

        class MockMol:
            atoms_bohr = (("H", np.zeros(3)), ("H", np.array([0.0, 0.0, 1.4])))

        coords_bohr = np.zeros((1, 2, 3)) + 1.0
        out = tmp_path / "bare.xyz"
        write_wigner_ensemble_xyz(out, MockMol(), coords_bohr)

        lines = out.read_text().strip().split("\n")
        tok = lines[2].split()
        # No velocity columns
        assert len(tok) == 4
        np.testing.assert_allclose(float(tok[1]), BOHR_TO_ANGSTROM, atol=1e-6)
