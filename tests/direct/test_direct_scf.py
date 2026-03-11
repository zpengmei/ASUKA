"""Tests for integral-direct SCF.

Verifies that the direct J/K path produces the same energy as the dense
AO-ERI path (exact agreement since both evaluate full 4-center integrals).
"""

from __future__ import annotations

import pytest
import numpy as np


def test_direct_jk_class_policy_parse():
    from asuka.hf import direct_jk as djk

    parsed = djk._parse_direct_jk_class_policy_env(
        "psss=warp_eri, ssss=fused, psdp=staged_warp_contract, noop=bogus"
    )

    assert parsed == {
        "psss": "staged_warp_eri",
        "ssss": "fused_jk",
        "psdp": "staged_warp_contract",
    }


def _cuda_available() -> bool:
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        from asuka.cueri.gpu import has_cuda_ext
        return has_cuda_ext()
    except Exception:
        return False


def _make_h2o(basis="sto-3g"):
    from asuka.frontend.molecule import Molecule

    return Molecule.from_atoms(
        [
            ("O", (0.0000000, 0.0000000, 0.1173470)),
            ("H", (0.0000000, 0.7572153, -0.4693878)),
            ("H", (0.0000000, -0.7572153, -0.4693878)),
        ],
        unit="angstrom",
        charge=0,
        spin=0,
        basis=basis,
        cart=True,
    )


def _make_lih():
    from asuka.frontend.molecule import Molecule

    return Molecule.from_atoms(
        [
            ("Li", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 1.6)),
        ],
        unit="angstrom",
        charge=0,
        spin=0,
        basis="sto-3g",
        cart=True,
    )


def _skip_no_cuda(test_func):
    test_func = pytest.mark.cuda(test_func)
    return pytest.mark.skipif(
        not _cuda_available(),
        reason="CUDA or cuERI ext not available",
    )(test_func)


@_skip_no_cuda
def test_direct_jk_single_iteration():
    """Verify single-iteration J/K values match dense (random D)."""
    import cupy as cp

    from asuka.frontend.one_electron import build_ao_basis_cart
    from asuka.integrals.int1e_cart import build_int1e_cart, nao_cart_from_basis
    from asuka.hf.dense_eri import build_ao_eri_dense
    from asuka.hf.dense_jk import dense_JK_from_eri_mat_D
    from asuka.hf.direct_jk import make_direct_jk_context, direct_JK

    mol = _make_h2o()
    ao_basis, _ = build_ao_basis_cart(mol, basis=mol.basis, expand_contractions=True)
    coords = np.asarray([xyz for _sym, xyz in mol.atoms_bohr], dtype=np.float64).reshape((-1, 3))
    charges = np.asarray([8, 1, 1], dtype=np.float64)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    nao = int(nao_cart_from_basis(ao_basis))

    # Random symmetric density
    rng = np.random.default_rng(42)
    D_np = rng.standard_normal((nao, nao))
    D_np = 0.5 * (D_np + D_np.T)
    D = cp.asarray(D_np, dtype=cp.float64)

    # Dense path
    dense = build_ao_eri_dense(ao_basis, backend="cuda", eps_ao=0.0)
    J_dense, K_dense = dense_JK_from_eri_mat_D(dense.eri_mat, D)

    # Direct path
    ctx = make_direct_jk_context(ao_basis, eps_schwarz=0.0)
    J_direct, K_direct = direct_JK(ctx, D)

    J_err = float(cp.max(cp.abs(J_dense - J_direct)).item())
    K_err = float(cp.max(cp.abs(K_dense - K_direct)).item())

    assert J_err < 1e-12, f"J mismatch: max|ΔJ| = {J_err:.2e}"
    assert K_err < 1e-12, f"K mismatch: max|ΔK| = {K_err:.2e}"


@_skip_no_cuda
def test_direct_fock_rhf_matches_jk(monkeypatch):
    """direct_fock_rhf must match h + J - 0.5*K for the same density."""
    import cupy as cp

    from asuka.frontend.one_electron import build_ao_basis_cart
    from asuka.integrals.int1e_cart import build_int1e_cart, nao_cart_from_basis
    from asuka.hf.direct_jk import direct_JK, direct_fock_rhf, make_direct_jk_context

    monkeypatch.setenv("ASUKA_DIRECT_FOCK_FUSED", "0")
    monkeypatch.delenv("ASUKA_DIRECT_FOCK_FUSED_ONLY", raising=False)
    monkeypatch.delenv("ASUKA_DIRECT_FOCK_FUSED_ENABLE", raising=False)
    monkeypatch.delenv("ASUKA_DIRECT_FOCK_FUSED_DISABLE", raising=False)

    mol = _make_h2o(basis="6-31g*")
    ao_basis, _ = build_ao_basis_cart(mol, basis=mol.basis, expand_contractions=True)
    coords = np.asarray([xyz for _sym, xyz in mol.atoms_bohr], dtype=np.float64).reshape((-1, 3))
    charges = np.asarray([8, 1, 1], dtype=np.float64)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    nao = int(nao_cart_from_basis(ao_basis))
    h = cp.asarray(int1e.hcore, dtype=cp.float64)

    rng = np.random.default_rng(123)
    D_np = rng.standard_normal((nao, nao))
    D_np = 0.5 * (D_np + D_np.T)
    D = cp.asarray(D_np, dtype=cp.float64)

    ctx = make_direct_jk_context(ao_basis, eps_schwarz=0.0)
    J, K = direct_JK(ctx, D, want_J=True, want_K=True)
    F_ref = h + J - 0.5 * K
    F_ref = 0.5 * (F_ref + F_ref.T)

    F = direct_fock_rhf(ctx, D, h)

    err = float(cp.max(cp.abs(F_ref - F)).item())
    assert err < 1e-12, f"Fock mismatch: max|ΔF|={err:.2e}"


@_skip_no_cuda
def test_direct_fock_rhf_fused_toggle_matches(monkeypatch):
    """Fused ERI->Fock path must match staged path (env toggle)."""
    import cupy as cp

    from asuka.frontend.one_electron import build_ao_basis_cart
    from asuka.integrals.int1e_cart import build_int1e_cart, nao_cart_from_basis
    from asuka.hf.direct_jk import direct_fock_rhf, make_direct_jk_context

    mol = _make_h2o(basis="6-31g*")
    ao_basis, _ = build_ao_basis_cart(mol, basis=mol.basis, expand_contractions=True)
    coords = np.asarray([xyz for _sym, xyz in mol.atoms_bohr], dtype=np.float64).reshape((-1, 3))
    charges = np.asarray([8, 1, 1], dtype=np.float64)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    nao = int(nao_cart_from_basis(ao_basis))
    h = cp.asarray(int1e.hcore, dtype=cp.float64)

    rng = np.random.default_rng(7)
    D_np = rng.standard_normal((nao, nao))
    D_np = 0.5 * (D_np + D_np.T)
    D = cp.asarray(D_np, dtype=cp.float64)

    ctx = make_direct_jk_context(ao_basis, eps_schwarz=0.0)

    # Staged-only (disable fused).
    monkeypatch.setenv("ASUKA_DIRECT_FOCK_FUSED", "0")
    stats_staged: dict = {}
    F_staged = direct_fock_rhf(ctx, D, h, stats=stats_staged)
    assert int(stats_staged.get("n_fused_calls", 0)) == 0

    # Fused enabled (force only psss so the test checks fused dispatch is actually taken).
    monkeypatch.setenv("ASUKA_DIRECT_FOCK_FUSED", "1")
    monkeypatch.setenv("ASUKA_DIRECT_FOCK_FUSED_ONLY", "psss")
    stats_fused: dict = {}
    F_fused = direct_fock_rhf(ctx, D, h, stats=stats_fused)
    assert int(stats_fused.get("n_fused_calls", 0)) > 0

    err = float(cp.max(cp.abs(F_staged - F_fused)).item())
    assert err < 1e-12, f"Fused vs staged Fock mismatch: max|ΔF|={err:.2e}"


@_skip_no_cuda
@pytest.mark.parametrize("fused_only", ["psss", "psds"])
def test_direct_jk_fused_toggle_matches(monkeypatch, fused_only):
    """Fused direct-JK specialized paths must match the staged path."""
    import cupy as cp

    from asuka.frontend.one_electron import build_ao_basis_cart
    from asuka.hf.direct_jk import direct_JK, make_direct_jk_context
    from asuka.integrals.int1e_cart import nao_cart_from_basis

    mol = _make_h2o(basis="6-31g*")
    ao_basis, _ = build_ao_basis_cart(mol, basis=mol.basis, expand_contractions=True)

    nao = int(nao_cart_from_basis(ao_basis))
    rng = np.random.default_rng(17)
    D_np = rng.standard_normal((nao, nao))
    D_np = 0.5 * (D_np + D_np.T)
    D = cp.asarray(D_np, dtype=cp.float64)

    ctx = make_direct_jk_context(ao_basis, eps_schwarz=0.0)

    monkeypatch.setenv("ASUKA_DIRECT_JK_FUSED", "0")
    monkeypatch.delenv("ASUKA_DIRECT_JK_FUSED_ONLY", raising=False)
    monkeypatch.delenv("ASUKA_DIRECT_JK_FUSED_ENABLE", raising=False)
    monkeypatch.delenv("ASUKA_DIRECT_JK_FUSED_DISABLE", raising=False)
    stats_staged: dict = {}
    J_staged, K_staged = direct_JK(ctx, D, want_J=True, want_K=True, stats=stats_staged)
    assert int(stats_staged.get("n_fused_calls", 0)) == 0

    monkeypatch.setenv("ASUKA_DIRECT_JK_FUSED", "1")
    monkeypatch.setenv("ASUKA_DIRECT_JK_FUSED_ONLY", fused_only)
    stats_fused: dict = {}
    J_fused, K_fused = direct_JK(ctx, D, want_J=True, want_K=True, stats=stats_fused)
    assert int(stats_fused.get("n_fused_calls", 0)) > 0

    err_j = float(cp.max(cp.abs(J_staged - J_fused)).item())
    err_k = float(cp.max(cp.abs(K_staged - K_fused)).item())
    assert err_j < 1e-12, f"Fused vs staged J mismatch for {fused_only}: max|ΔJ|={err_j:.2e}"
    assert err_k < 1e-12, f"Fused vs staged K mismatch for {fused_only}: max|ΔK|={err_k:.2e}"


@_skip_no_cuda
def test_direct_jk_vs_dense_lih():
    """Direct SCF must match dense SCF energy on LiH/STO-3G."""
    from asuka.frontend.scf import run_hf_df

    mol = _make_lih()

    dense_out = run_hf_df(mol, two_e_backend="dense", dense_eps_ao=0.0)
    direct_out = run_hf_df(mol, two_e_backend="direct", eps_schwarz=0.0)

    e_dense = dense_out.scf.e_tot
    e_direct = direct_out.scf.e_tot

    assert dense_out.scf.converged, f"Dense SCF did not converge: E={e_dense}"
    assert direct_out.scf.converged, f"Direct SCF did not converge: E={e_direct}"

    err = abs(e_dense - e_direct)
    assert err < 1e-10, f"Direct vs dense energy mismatch: |ΔE| = {err:.2e} (dense={e_dense}, direct={e_direct})"


@_skip_no_cuda
def test_direct_jk_vs_dense_h2o():
    """Direct SCF must match dense SCF energy on H2O/STO-3G."""
    from asuka.frontend.scf import run_hf_df

    mol = _make_h2o()

    dense_out = run_hf_df(mol, two_e_backend="dense", dense_eps_ao=0.0)
    direct_out = run_hf_df(mol, two_e_backend="direct", eps_schwarz=0.0)

    e_dense = dense_out.scf.e_tot
    e_direct = direct_out.scf.e_tot

    assert dense_out.scf.converged
    assert direct_out.scf.converged

    err = abs(e_dense - e_direct)
    assert err < 1e-10, f"Direct vs dense energy mismatch: |ΔE| = {err:.2e}"


@_skip_no_cuda
def test_direct_jk_vs_dense_h2o_631g():
    """Direct vs dense on H2O/6-31G* (includes d-functions)."""
    from asuka.frontend.scf import run_hf_df

    mol = _make_h2o(basis="6-31g*")

    dense_out = run_hf_df(mol, two_e_backend="dense", dense_eps_ao=0.0)
    direct_out = run_hf_df(mol, two_e_backend="direct", eps_schwarz=0.0)

    e_dense = dense_out.scf.e_tot
    e_direct = direct_out.scf.e_tot

    assert dense_out.scf.converged
    assert direct_out.scf.converged

    err = abs(e_dense - e_direct)
    assert err < 1e-10, f"Direct vs dense energy mismatch: |ΔE| = {err:.2e}"


@_skip_no_cuda
def test_direct_jk_with_schwarz_screening():
    """Direct J/K with Schwarz screening should still match dense closely."""
    from asuka.frontend.scf import run_hf_df

    mol = _make_h2o()

    dense_out = run_hf_df(mol, two_e_backend="dense", dense_eps_ao=0.0)
    direct_out = run_hf_df(mol, two_e_backend="direct", eps_schwarz=1e-12)

    e_dense = dense_out.scf.e_tot
    e_direct = direct_out.scf.e_tot

    assert direct_out.scf.converged

    err = abs(e_dense - e_direct)
    assert err < 1e-10, f"Direct (screened) vs dense mismatch: |ΔE| = {err:.2e}"


def _make_li():
    """Li atom (doublet) for UHF/ROHF tests."""
    from asuka.frontend.molecule import Molecule

    return Molecule.from_atoms(
        [("Li", (0.0, 0.0, 0.0))],
        unit="angstrom",
        charge=0,
        spin=1,
        basis="sto-3g",
        cart=True,
    )


def _make_h2():
    """H2 for direct->CASCI/CASSCF smoke tests."""
    from asuka.frontend.molecule import Molecule

    return Molecule.from_atoms(
        [
            ("H", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 0.74)),
        ],
        unit="angstrom",
        charge=0,
        spin=0,
        basis="sto-3g",
        cart=True,
    )


@_skip_no_cuda
def test_uhf_direct_vs_dense():
    """UHF direct must match dense energy on Li/STO-3G."""
    from asuka.frontend.scf import run_hf_df

    mol = _make_li()

    dense_out = run_hf_df(mol, method="uhf", two_e_backend="dense", dense_eps_ao=0.0)
    direct_out = run_hf_df(mol, method="uhf", two_e_backend="direct", eps_schwarz=0.0)

    assert dense_out.scf.converged, f"Dense UHF did not converge"
    assert direct_out.scf.converged, f"Direct UHF did not converge"

    err = abs(dense_out.scf.e_tot - direct_out.scf.e_tot)
    assert err < 1e-10, f"UHF direct vs dense: |ΔE| = {err:.2e}"


@_skip_no_cuda
def test_rohf_direct_vs_dense():
    """ROHF direct must match dense energy on Li/STO-3G."""
    from asuka.frontend.scf import run_hf_df

    mol = _make_li()

    dense_out = run_hf_df(mol, method="rohf", two_e_backend="dense", dense_eps_ao=0.0)
    direct_out = run_hf_df(mol, method="rohf", two_e_backend="direct", eps_schwarz=0.0)

    assert dense_out.scf.converged, f"Dense ROHF did not converge"
    assert direct_out.scf.converged, f"Direct ROHF did not converge"

    err = abs(dense_out.scf.e_tot - direct_out.scf.e_tot)
    assert err < 1e-10, f"ROHF direct vs dense: |ΔE| = {err:.2e}"


@_skip_no_cuda
def test_want_J_only():
    """direct_JK with want_K=False should return J and K=None."""
    import cupy as cp
    from asuka.frontend.one_electron import build_ao_basis_cart
    from asuka.integrals.int1e_cart import nao_cart_from_basis
    from asuka.hf.direct_jk import make_direct_jk_context, direct_JK

    mol = _make_h2o()
    ao_basis, _ = build_ao_basis_cart(mol, basis=mol.basis, expand_contractions=True)
    nao = int(nao_cart_from_basis(ao_basis))

    rng = np.random.default_rng(42)
    D_np = rng.standard_normal((nao, nao))
    D_np = 0.5 * (D_np + D_np.T)
    D = cp.asarray(D_np, dtype=cp.float64)

    ctx = make_direct_jk_context(ao_basis, eps_schwarz=0.0)

    # Full build
    J_full, K_full = direct_JK(ctx, D, want_J=True, want_K=True)

    # J-only
    J_only, K_none = direct_JK(ctx, D, want_J=True, want_K=False)

    assert K_none is None, "K should be None when want_K=False"
    j_err = float(cp.max(cp.abs(J_full - J_only)).item())
    assert j_err < 1e-12, f"J mismatch with want_K=False: {j_err:.2e}"


@_skip_no_cuda
def test_cpu_slab_forced():
    """Forcing CPU slabs (gpu_task_budget_bytes=0) must give same J/K."""
    import cupy as cp
    from asuka.frontend.one_electron import build_ao_basis_cart
    from asuka.integrals.int1e_cart import nao_cart_from_basis
    from asuka.hf.direct_jk import make_direct_jk_context, direct_JK

    mol = _make_h2o()
    ao_basis, _ = build_ao_basis_cart(mol, basis=mol.basis, expand_contractions=True)
    nao = int(nao_cart_from_basis(ao_basis))

    rng = np.random.default_rng(42)
    D_np = rng.standard_normal((nao, nao))
    D_np = 0.5 * (D_np + D_np.T)
    D = cp.asarray(D_np, dtype=cp.float64)

    # GPU-resident slabs (default)
    ctx_gpu = make_direct_jk_context(ao_basis, eps_schwarz=0.0)
    J_gpu, K_gpu = direct_JK(ctx_gpu, D)

    # Force CPU slabs
    ctx_cpu = make_direct_jk_context(ao_basis, eps_schwarz=0.0, gpu_task_budget_bytes=0)
    assert all(not s.gpu_resident for s in ctx_cpu.slabs), "Slabs should be CPU-resident"
    J_cpu, K_cpu = direct_JK(ctx_cpu, D)

    j_err = float(cp.max(cp.abs(J_gpu - J_cpu)).item())
    k_err = float(cp.max(cp.abs(K_gpu - K_cpu)).item())
    assert j_err < 1e-12, f"J mismatch GPU vs CPU slab: {j_err:.2e}"
    assert k_err < 1e-12, f"K mismatch GPU vs CPU slab: {k_err:.2e}"


@_skip_no_cuda
def test_shape_rejection():
    """direct_JK must reject D with wrong shape."""
    import cupy as cp
    from asuka.frontend.one_electron import build_ao_basis_cart
    from asuka.integrals.int1e_cart import nao_cart_from_basis
    from asuka.hf.direct_jk import make_direct_jk_context, direct_JK

    mol = _make_h2o()
    ao_basis, _ = build_ao_basis_cart(mol, basis=mol.basis, expand_contractions=True)
    nao = int(nao_cart_from_basis(ao_basis))
    ctx = make_direct_jk_context(ao_basis, eps_schwarz=0.0)

    # 1D array (wrong ndim)
    D_1d = cp.zeros(nao * nao, dtype=cp.float64)
    with pytest.raises(ValueError, match="must be"):
        direct_JK(ctx, D_1d)

    # Wrong 2D shape
    D_wrong = cp.zeros((nao + 1, nao), dtype=cp.float64)
    with pytest.raises(ValueError, match="must be"):
        direct_JK(ctx, D_wrong)


@_skip_no_cuda
def test_direct_scf_to_casci_dense_gpu_smoke():
    from asuka.frontend.scf import run_hf_df
    from asuka.mcscf.casci import run_casci

    scf_out = run_hf_df(_make_h2(), method="rhf", backend="cuda", two_e_backend="direct", eps_schwarz=0.0)
    cas = run_casci(
        scf_out,
        ncore=0,
        ncas=2,
        nelecas=2,
        backend="cuda",
        df=False,
        matvec_backend="cuda_eri_mat",
    )
    assert float(cas.e_tot) < 0.0


@_skip_no_cuda
def test_direct_scf_to_casci_cpu_rejected():
    from asuka.frontend.scf import run_hf_df
    from asuka.mcscf.casci import run_casci

    scf_out = run_hf_df(_make_h2(), method="rhf", backend="cuda", two_e_backend="direct", eps_schwarz=0.0)
    with pytest.raises(NotImplementedError):
        run_casci(scf_out, ncore=0, ncas=2, nelecas=2, backend="cpu", df=False)


@_skip_no_cuda
@pytest.mark.parametrize("optimizer", ["jacobi", "lbfgs", "ah", "1step"])
def test_direct_scf_to_casscf_smoke(optimizer: str):
    from asuka.frontend.scf import run_hf_df
    from asuka.mcscf import run_casscf

    scf_out = run_hf_df(_make_h2(), method="rhf", backend="cuda", two_e_backend="direct", eps_schwarz=0.0)
    mc = run_casscf(
        scf_out,
        ncore=0,
        ncas=2,
        nelecas=2,
        backend="cuda",
        df=False,
        orbital_optimizer=str(optimizer),
        max_cycle_macro=1,
        tol=1e-9,
        conv_tol_grad=1e-5,
        matvec_backend="cuda_eri_mat",
    )
    assert float(mc.e_tot) < 0.0


def test_build_provider_newton_eris_projects_jpc_kpc_in_mo_basis():
    from asuka.mcscf.newton_df import build_provider_newton_eris

    class _FakeProvider:
        def probe_array(self):
            return np.zeros((2, 2), dtype=np.float64)

        def build_pq_uv(self, C_mo, C_act):
            nmo = int(C_mo.shape[1])
            ncas = int(C_act.shape[1])
            return np.zeros((nmo * nmo, ncas * ncas), dtype=np.float64)

        def build_pu_qv(self, C_mo, C_act):
            nmo = int(C_mo.shape[1])
            ncas = int(C_act.shape[1])
            return np.zeros((nmo * ncas, nmo * ncas), dtype=np.float64)

        def _jk_pair(self):
            # Non-diagonal AO matrices: AO diagonal and MO diagonal differ.
            J = np.asarray([[1.0, 0.5], [0.5, 2.0]], dtype=np.float64)
            K = np.asarray([[0.8, -0.2], [-0.2, 0.3]], dtype=np.float64)
            return J, K

        def jk(self, D, *, want_J=True, want_K=True):
            _ = D
            J, K = self._jk_pair()
            return (J if want_J else None), (K if want_K else None)

        def jk_multi2(self, Da, Db, *, want_J=True, want_K=True):
            _ = Da, Db
            J, K = self._jk_pair()
            return (J if want_J else None), (K if want_K else None), (J if want_J else None), (K if want_K else None)

    s2 = np.sqrt(0.5)
    mo = np.asarray([[s2, s2], [s2, -s2]], dtype=np.float64)
    provider = _FakeProvider()

    eris = build_provider_newton_eris(provider, mo, ncore=1, ncas=1)
    J, K = provider._jk_pair()
    j_expected = np.diag(mo.T @ J @ mo)
    k_expected = np.diag(mo.T @ K @ mo)

    assert np.allclose(eris.j_pc[:, 0], j_expected, atol=1e-12, rtol=1e-12)
    assert np.allclose(eris.k_pc[:, 0], k_expected, atol=1e-12, rtol=1e-12)


@_skip_no_cuda
def test_direct_scf_to_casscf_nuc_grad_smoke():
    from asuka.frontend.scf import run_hf_df
    from asuka.mcscf import casscf_nuc_grad, run_casscf

    scf_out = run_hf_df(
        _make_h2(),
        method="rhf",
        backend="cuda",
        two_e_backend="direct",
        eps_schwarz=0.0,
        max_cycle=50,
        conv_tol=1e-12,
        conv_tol_dm=1e-10,
    )
    assert bool(getattr(scf_out.scf, "converged", False))

    mc = run_casscf(
        scf_out,
        ncore=0,
        ncas=2,
        nelecas=2,
        backend="cuda",
        df=False,
        nroots=1,
        max_cycle_macro=20,
        tol=1e-8,
        conv_tol_grad=1e-6,
        orbital_optimizer="lbfgs",
        matvec_backend="cuda_eri_mat",
    )
    assert bool(getattr(mc, "converged", False))

    g_direct = casscf_nuc_grad(
        scf_out,
        mc,
        backend="direct",
        direct_eri_deriv_backend="cpu",
    )
    g_auto = casscf_nuc_grad(
        scf_out,
        mc,
        backend="auto",
        direct_eri_deriv_backend="cpu",
    )
    assert g_direct.grad.shape == (int(scf_out.mol.natm), 3)
    assert bool(np.isfinite(g_direct.grad).all())
    np.testing.assert_allclose(
        np.asarray(g_auto.grad, dtype=np.float64),
        np.asarray(g_direct.grad, dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )


@_skip_no_cuda
def test_direct_scf_to_casscf_nuc_grad_per_root_smoke():
    from asuka.frontend.scf import run_hf_df
    from asuka.mcscf import casscf_nuc_grad_per_root, run_casscf

    scf_out = run_hf_df(
        _make_h2(),
        method="rhf",
        backend="cuda",
        two_e_backend="direct",
        eps_schwarz=0.0,
        max_cycle=50,
        conv_tol=1e-12,
        conv_tol_dm=1e-10,
    )
    assert bool(getattr(scf_out.scf, "converged", False))

    mc = run_casscf(
        scf_out,
        ncore=0,
        ncas=2,
        nelecas=2,
        backend="cuda",
        df=False,
        nroots=2,
        root_weights=[0.5, 0.5],
        max_cycle_macro=20,
        tol=1e-8,
        conv_tol_grad=1e-6,
        orbital_optimizer="lbfgs",
        matvec_backend="cuda_eri_mat",
    )
    assert bool(getattr(mc, "converged", False))

    out = casscf_nuc_grad_per_root(
        scf_out,
        mc,
        backend="direct",
        direct_eri_deriv_backend="cpu",
        z_maxiter=120,
    )
    grads = np.asarray(out.grads, dtype=np.float64)
    assert grads.shape == (2, int(scf_out.mol.natm), 3)
    assert bool(np.isfinite(grads).all())
    assert np.asarray(out.grad_sa, dtype=np.float64).shape == (int(scf_out.mol.natm), 3)


@_skip_no_cuda
def test_direct_scf_to_casscf_nuc_grad_per_root_fd_component():
    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import run_hf_df
    from asuka.mcscf import casscf_nuc_grad_per_root, run_casscf
    from asuka.mcscf.state_average import ci_as_list
    from asuka.mrci.common import assign_roots_by_overlap

    def _run_point(r_h_ang: float):
        mol = Molecule.from_atoms(
            [
                ("H", (0.0, 0.0, 0.0)),
                ("H", (0.0, 0.0, float(r_h_ang))),
            ],
            unit="angstrom",
            charge=0,
            spin=0,
            basis="sto-3g",
            cart=True,
        )
        scf_out = run_hf_df(
            mol,
            method="rhf",
            backend="cuda",
            two_e_backend="direct",
            eps_schwarz=0.0,
            max_cycle=60,
            conv_tol=1e-12,
            conv_tol_dm=1e-10,
        )
        assert bool(getattr(scf_out.scf, "converged", False))
        mc = run_casscf(
            scf_out,
            ncore=0,
            ncas=2,
            nelecas=2,
            backend="cuda",
            df=False,
            nroots=2,
            root_weights=[0.5, 0.5],
            max_cycle_macro=40,
            tol=1e-9,
            conv_tol_grad=1e-7,
            orbital_optimizer="lbfgs",
            matvec_backend="cuda_eri_mat",
        )
        assert bool(getattr(mc, "converged", False))
        out = casscf_nuc_grad_per_root(
            scf_out,
            mc,
            backend="direct",
            direct_eri_deriv_backend="cpu",
            z_maxiter=300,
            z_tol=1e-11,
        )
        return mc, out

    # Component: atom 1, z
    ia, xyz = 1, 2
    h_bohr = 1.0e-3
    h_ang = h_bohr * 0.529177210903
    r0 = 0.74

    mc0, out0 = _run_point(r0)
    mcp, _outp = _run_point(r0 + h_ang)
    mcm, _outm = _run_point(r0 - h_ang)

    ci0 = [np.asarray(c, dtype=np.float64).ravel() for c in ci_as_list(mc0.ci, nroots=2)]
    cip = [np.asarray(c, dtype=np.float64).ravel() for c in ci_as_list(mcp.ci, nroots=2)]
    cim = [np.asarray(c, dtype=np.float64).ravel() for c in ci_as_list(mcm.ci, nroots=2)]

    ov_p = np.zeros((2, 2), dtype=np.float64)
    ov_m = np.zeros((2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            ov_p[i, j] = abs(float(np.dot(ci0[i], cip[j]))) ** 2
            ov_m[i, j] = abs(float(np.dot(ci0[i], cim[j]))) ** 2
    map_p = assign_roots_by_overlap(ov_p)
    map_m = assign_roots_by_overlap(ov_m)

    e_p = np.asarray(mcp.e_roots, dtype=np.float64)
    e_m = np.asarray(mcm.e_roots, dtype=np.float64)
    grads = np.asarray(out0.grads, dtype=np.float64)
    errs = []
    for k in range(2):
        fd = float((e_p[int(map_p[k])] - e_m[int(map_m[k])]) / (2.0 * h_bohr))
        ana = float(grads[k, ia, xyz])
        errs.append(abs(ana - fd))

    assert max(errs) < 1.0e-5, f"direct per-root FD mismatch too large: max_err={max(errs):.3e}"
