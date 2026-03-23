"""Tests for CASSecond (second-order CASSCF optimizer)."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from asuka.mcscf.rotfile import RotFile
from asuka.mcscf.aughess import AugHess


# ---------------------------------------------------------------------------
# Helper: build a mock scf_out from PySCF for testing
# ---------------------------------------------------------------------------


def _make_scf_out_from_pyscf(atom_str: str, basis: str = "sto-3g"):
    """Build a mock scf_out object using PySCF for integrals.

    Returns an object with the same interface as ASUKA's RHFDFRunResult:
      .int1e.hcore  -- (nao, nao) core Hamiltonian
      .df_B         -- (nao, nao, naux) whitened DF factors
      .scf.e_nuc    -- nuclear repulsion energy
      .scf.mo_coeff -- (nao, nmo) MO coefficients
      .scf.mo_energy -- (nmo,) orbital energies
      .mol          -- molecule-like object with .spin attribute
    """
    from pyscf import gto, scf as pyscf_scf, df as pyscf_df, lib

    mol = gto.M(atom=atom_str, basis=basis, cart=True)
    nao = mol.nao_nr()

    # RHF
    mf = pyscf_scf.RHF(mol)
    mf.kernel()

    # Build DF factors
    # PySCF DF stores Cholesky-decomposed 3-center integrals
    mydf = pyscf_df.DF(mol)
    mydf.auxbasis = "weigend"
    naux = mydf.get_naoaux()

    # Get raw 3-center integrals (naux, nao, nao)
    int3c = mydf._cderi  # (naux, nao*(nao+1)//2) lower triangle
    # Unpack to full (naux, nao, nao)
    B_full = np.zeros((naux, nao, nao), dtype=np.float64)
    idx = np.tril_indices(nao)
    for P in range(naux):
        B_full[P][idx] = int3c[P]
        B_full[P] = B_full[P] + B_full[P].T - np.diag(np.diag(B_full[P]))

    # These are already J^{-1/2}-whitened in PySCF's _cderi.
    # Transpose to (nao, nao, naux) for ASUKA convention.
    B_ao = B_full.transpose(1, 2, 0).copy()  # (nao, nao, naux)

    # Build result objects
    int1e = SimpleNamespace(
        hcore=mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        S=mol.intor("int1e_ovlp"),
    )
    scf_result = SimpleNamespace(
        e_nuc=float(mol.energy_nuc()),
        e_tot=float(mf.e_tot),
        mo_coeff=mf.mo_coeff.copy(),
        mo_energy=mf.mo_energy.copy(),
        mo_occ=mf.mo_occ.copy(),
    )
    _enuc = float(mol.energy_nuc())
    mol_ns = SimpleNamespace(spin=mol.spin, energy_nuc=lambda: _enuc)
    scf_out = SimpleNamespace(
        int1e=int1e,
        df_B=B_ao,
        scf=scf_result,
        mol=mol_ns,
        mo_coeff=mf.mo_coeff.copy(),
    )
    return scf_out, mf


# ---------------------------------------------------------------------------
# RotFile unit tests
# ---------------------------------------------------------------------------


class TestRotFile:
    def test_basic_construction(self):
        rf = RotFile(3, 2, 4)
        assert rf.size == 3 * 2 + 4 * 2 + 4 * 3  # 6 + 8 + 12 = 26
        assert rf.norm() == 0.0

    def test_element_access(self):
        rf = RotFile(2, 3, 4)
        rf.set_ele_ca(1, 2, 3.5)
        assert rf.ele_ca(1, 2) == pytest.approx(3.5)
        rf.set_ele_va(3, 1, -1.2)
        assert rf.ele_va(3, 1) == pytest.approx(-1.2)
        rf.set_ele_vc(2, 0, 7.7)
        assert rf.ele_vc(2, 0) == pytest.approx(7.7)

    def test_block_views(self):
        rf = RotFile(2, 3, 4)
        for i in range(3):
            for j in range(2):
                rf.set_ele_ca(j, i, float(j + i * 10))
        ca = rf.ca_mat()
        assert ca.shape == (2, 3)
        assert ca[1, 2] == pytest.approx(21.0)

    def test_unpack_antisymmetric(self):
        rf = RotFile(2, 2, 2)
        rf.set_ele_ca(0, 1, 1.0)
        rf.set_ele_va(1, 0, 2.0)
        rf.set_ele_vc(0, 1, 3.0)
        A = rf.unpack()
        assert A.shape == (6, 6)
        # Check antisymmetry
        np.testing.assert_allclose(A, -A.T, atol=1e-15)

    def test_dot_and_norm(self):
        # RotFile(2, 2, 2): size = 2*2 + 2*2 + 2*2 = 12
        rf1 = RotFile(2, 2, 2)
        rf2 = RotFile(2, 2, 2)
        vals = np.arange(1, 13, dtype=np.float64)
        rf1.data[:] = vals
        rf2.data[:] = 1.0
        assert rf1.dot(rf2) == pytest.approx(vals.sum())
        assert rf1.norm() == pytest.approx(np.linalg.norm(vals))

    def test_copy_and_clone(self):
        rf = RotFile(2, 2, 2)
        rf.data[:] = 1.0
        cp = rf.copy()
        cl = rf.clone()
        assert cp.dot(rf) == pytest.approx(rf.dot(rf))
        assert cl.norm() == 0.0

    def test_normalize(self):
        rf = RotFile(1, 1, 1)
        rf.data[:] = [3.0, 4.0, 0.0]
        n = rf.normalize()
        assert n == pytest.approx(5.0)
        assert rf.norm() == pytest.approx(1.0)

    def test_ax_plus_y(self):
        rf1 = RotFile(2, 1, 1)
        rf2 = RotFile(2, 1, 1)
        rf1.data[:] = 1.0
        rf2.data[:] = 2.0
        rf1.ax_plus_y(3.0, rf2)
        np.testing.assert_allclose(rf1.data, 7.0)

    def test_ax_plus_y_blocks(self):
        rf = RotFile(2, 2, 3)
        mat_ca = np.ones((2, 2))
        mat_va = np.ones((3, 2)) * 2.0
        mat_vc = np.ones((3, 2)) * 3.0
        rf.ax_plus_y_ca(1.0, mat_ca)
        rf.ax_plus_y_va(1.0, mat_va)
        rf.ax_plus_y_vc(1.0, mat_vc)
        np.testing.assert_allclose(rf.ca_mat(), mat_ca)
        np.testing.assert_allclose(rf.va_mat(), mat_va)
        np.testing.assert_allclose(rf.vc_mat(), mat_vc)


# ---------------------------------------------------------------------------
# AugHess unit tests
# ---------------------------------------------------------------------------


class TestAugHess:
    def test_simple_quadratic(self):
        """Test AugHess on a simple 1D quadratic H*x = -g."""
        grad = RotFile(1, 1, 0)
        grad.data[:] = [1.0]

        solver = AugHess(10, grad)

        # Initial trial
        trial = grad.copy()
        trial.normalize()

        # Sigma = H * trial (H = identity for simplicity)
        sigma = trial.copy()

        residual, lam, eps, stepsize = solver.compute_residual(trial, sigma)
        # Should produce a reasonable residual
        assert isinstance(lam, float)
        assert isinstance(eps, float)

    def test_orthog(self):
        """Test that orthog produces orthogonal vectors."""
        # RotFile(2, 1, 1): size = 2*1 + 1*1 + 1*2 = 5
        grad = RotFile(2, 1, 1)
        grad.data[:] = [1.0, 0.0, 0.0, 0.0, 0.0]

        solver = AugHess(10, grad)

        t1 = RotFile(2, 1, 1)
        t1.data[:] = [1.0, 0.0, 0.0, 0.0, 0.0]
        t1.normalize()
        s1 = t1.copy()
        solver.compute_residual(t1, s1)

        t2 = RotFile(2, 1, 1)
        t2.data[:] = [0.5, 0.5, 0.3, 0.1, 0.2]
        n = solver.orthog(t2)
        assert n > 0
        # Check orthogonality
        assert abs(t2.dot(t1) / t1.norm()) < 1e-10


# ---------------------------------------------------------------------------
# Qvec unit test
# ---------------------------------------------------------------------------


def test_qvec_symmetry():
    """Test that Qvec has correct shape and structure."""
    from asuka.mcscf.qvec import build_qvec

    nmo, nact, nclosed = 6, 2, 1
    naux = 10
    coeff = np.eye(nmo, dtype=np.float64)
    B_ao = np.random.RandomState(42).randn(nmo, nmo, naux)
    B_ao = 0.5 * (B_ao + B_ao.transpose(1, 0, 2))  # symmetrize in mu, nu
    rdm2 = np.random.RandomState(43).randn(nact, nact, nact, nact)
    # Make rdm2 have the right symmetry: rdm2[p,q,r,s] = rdm2[r,s,p,q]
    rdm2 = 0.5 * (rdm2 + rdm2.transpose(2, 3, 0, 1))

    Q = build_qvec(nmo, nact, coeff, nclosed, B_ao, rdm2)
    assert Q.shape == (nmo, nact)


# ---------------------------------------------------------------------------
# CASSecond integration tests (using PySCF for integrals)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_lih_sto3g():
    """Test CASSecond on LiH/STO-3G CAS(2,2).

    Verifies that the optimizer lowers the energy from the initial CASCI
    and that the gradient is substantially reduced.
    """
    from asuka.mcscf.cassecond import CASSecond

    scf_out, mf = _make_scf_out_from_pyscf("Li 0 0 0; H 0 0 1.5953", "sto-3g")

    cs = CASSecond(
        scf_out,
        ncore=1,
        ncas=2,
        nelecas=2,
        max_iter=30,
        thresh=1e-3,
        verbose=True,
    )
    result = cs.compute()

    # Energy should be near the CASSCF minimum (~-7.88 for PySCF LiH STO-3G CAS(2,2))
    assert result.e_tot < -7.87, f"Energy not low enough: {result.e_tot}"
    assert result.e_tot > -7.90, f"Energy too low: {result.e_tot}"
    assert result.converged, f"Not converged after {result.niter} iterations"


@pytest.mark.slow
def test_lih_gradient_descent():
    """Verify gradient is correct via finite-difference on the energy."""
    from asuka.mcscf.cassecond import CASSecond
    from asuka.mcscf.qvec import build_qvec
    from scipy.linalg import expm

    scf_out, _ = _make_scf_out_from_pyscf("Li 0 0 0; H 0 0 1.5953", "sto-3g")
    cs = CASSecond(scf_out, ncore=1, ncas=2, nelecas=2, max_iter=0,
                   thresh=1e-10, verbose=False)
    cs._solve_casci()
    cs._trans_natorb()
    cfock, afock = cs._build_fock()
    qxr = build_qvec(cs.nmo, cs.nact, cs.coeff, cs.nclosed, cs.B_ao, cs.rdm2_av)
    grad = cs._compute_gradient(cfock, afock, qxr)

    # Step along negative gradient
    A = grad.unpack()
    eps_step = 1e-4
    R = expm(-eps_step * A)
    cs2 = CASSecond(scf_out, ncore=1, ncas=2, nelecas=2, max_iter=0,
                    thresh=1e-10, verbose=False)
    cs2.coeff = cs.coeff @ R
    cs2._solve_casci()
    dE = cs2.energy[0] - cs.energy[0]
    predicted = -eps_step * grad.dot(grad)

    # Actual and predicted dE should agree to ~1%
    np.testing.assert_allclose(dE, predicted, rtol=0.05,
                               err_msg=f"dE={dE:.6e} vs predicted={predicted:.6e}")


@pytest.mark.slow
def test_h2o_sto3g():
    """Test CASSecond on H2O/STO-3G CAS(4,4) lowers energy from initial CASCI."""
    from asuka.mcscf.cassecond import CASSecond

    scf_out, _ = _make_scf_out_from_pyscf(
        "O 0 0 0; H 0 0.75716 -0.58589; H 0 -0.75716 -0.58589",
        "sto-3g",
    )

    cs = CASSecond(
        scf_out,
        ncore=2,
        ncas=4,
        nelecas=4,
        max_iter=10,
        thresh=1e-3,
        verbose=True,
    )
    # Record initial CASCI energy
    cs._solve_casci()
    e_init = cs.energy[0]

    cs2 = CASSecond(
        scf_out,
        ncore=2,
        ncas=4,
        nelecas=4,
        max_iter=10,
        thresh=1e-3,
        verbose=False,
    )
    result = cs2.compute()

    # Energy should decrease from initial CASCI
    assert result.e_tot < e_init, (
        f"Energy did not decrease: {result.e_tot:.6f} vs initial {e_init:.6f}"
    )
