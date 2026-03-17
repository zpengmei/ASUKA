"""Compare ASUKA direct-SCF (4c ERI) SA-CASSCF against PySCF for
energy, SA gradient, per-root gradient, and NACV.

Both codes use exact 4-center ERIs so results should agree to near
machine precision (limited only by CASSCF convergence tolerance and
CI solver differences).

Molecule: LiH / STO-3G, CAS(2,2), SA-2 [0.5, 0.5]
"""

from __future__ import annotations

import numpy as np
import pytest


def _best_signed_diff(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Sign-aligned residual for NACV phase ambiguity."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff_pos = a - b
    diff_neg = a + b
    norm_pos = float(np.linalg.norm(diff_pos))
    norm_neg = float(np.linalg.norm(diff_neg))
    if norm_pos <= norm_neg:
        return diff_pos, norm_pos
    return diff_neg, norm_neg


# -- shared fixtures --------------------------------------------------------

_ATOMS = [("Li", (0.0, 0.0, 0.0)), ("H", (0.2, 0.1, 3.0))]
_BASIS = "sto-3g"
_NCORE, _NCAS, _NELECAS, _NROOTS = 1, 2, 2, 2


@pytest.fixture(scope="module")
def asuka_results():
    """Run ASUKA HF(direct) + SA-CASSCF + gradient + per-root gradient + NACV."""
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.zeros((1,), dtype=cp.float64)
    except Exception:
        pytest.skip("CuPy present but no CUDA device")

    try:
        from asuka.hf.direct_jk import make_direct_jk_context  # noqa: F401
    except Exception:
        pytest.skip("direct_jk extension unavailable")

    from asuka.frontend import Molecule, run_hf
    from asuka.mcscf import run_casscf
    from asuka.mcscf.nac._dense import sacasscf_nonadiabatic_couplings_dense
    from asuka.mcscf.nuc_grad import casscf_nuc_grad, casscf_nuc_grad_per_root

    mol = Molecule.from_atoms(_ATOMS, unit="Bohr", basis=_BASIS, cart=True)

    scf_out = run_hf(
        mol,
        method="rhf",
        backend="cuda",
        two_e_backend="direct",
        max_cycle=100,
        conv_tol=1e-12,
        conv_tol_dm=1e-10,
    )
    assert bool(getattr(scf_out.scf, "converged", False)), "ASUKA HF did not converge"

    mc = run_casscf(
        scf_out,
        ncore=_NCORE,
        ncas=_NCAS,
        nelecas=_NELECAS,
        nroots=_NROOTS,
        root_weights=[0.5, 0.5],
        backend="cuda",
        max_cycle_macro=50,
        conv_tol=1e-10,
    )
    assert bool(getattr(mc, "converged", False)), "ASUKA CASSCF did not converge"

    grad_sa = casscf_nuc_grad(scf_out, mc)

    grad_pr = casscf_nuc_grad_per_root(scf_out, mc)

    nacv = sacasscf_nonadiabatic_couplings_dense(
        scf_out,
        mc,
        pairs=[(0, 1)],
        mult_ediff=True,
        use_etfs=False,
        backend="cuda",
        response_term="split_orbfd",
        z_tol=1e-10,
        z_maxiter=200,
    )

    return {
        "e_hf": float(scf_out.scf.e_tot),
        "e_states": np.asarray(mc.e_roots, dtype=np.float64),
        "e_tot": float(mc.e_tot),
        "grad_sa": np.asarray(grad_sa.grad, dtype=np.float64),
        "grads_per_root": np.asarray(grad_pr.grads, dtype=np.float64),
        "nacv_01": np.asarray(nacv[0, 1], dtype=np.float64),
    }


@pytest.fixture(scope="module")
def pyscf_results():
    """Run PySCF RHF + SA-CASSCF + gradient + NACV (exact 4c ERIs)."""
    gto = pytest.importorskip("pyscf.gto")
    pyscf_scf = pytest.importorskip("pyscf.scf")
    pyscf_mcscf = pytest.importorskip("pyscf.mcscf")
    pyscf_fci = pytest.importorskip("pyscf.fci")

    from pyscf import fci, gto, mcscf, scf  # noqa: F811

    mol = gto.M(
        atom="Li 0 0 0; H 0.2 0.1 3.0",
        unit="Bohr",
        basis="sto-3g",
        cart=True,
        spin=0,
        verbose=0,
    )

    mf = scf.RHF(mol).run(conv_tol=1e-12)
    assert mf.converged, "PySCF HF did not converge"

    solver = fci.direct_spin0.FCI(mol)
    solver.nroots = _NROOTS
    mc = mcscf.CASSCF(mf, _NCAS, _NELECAS)
    mc.conv_tol = 1e-10
    mc.fcisolver = solver
    mc = mc.state_average_([0.5, 0.5]).run()
    assert mc.converged, "PySCF CASSCF did not converge"

    # SA gradient
    grad_sa = mc.nuc_grad_method().kernel()

    # Per-root gradients
    mc_scanner = mc.nuc_grad_method().as_scanner()
    grads_per_root = []
    for iroot in range(_NROOTS):
        mc_scanner.state = iroot
        g = mc_scanner.kernel()
        grads_per_root.append(g)
    grads_per_root = np.array(grads_per_root)

    # NACV
    nacv_01 = np.asarray(
        mc.nac_method().kernel(state=(0, 1), use_etfs=False, mult_ediff=True),
        dtype=np.float64,
    )

    return {
        "e_hf": float(mf.e_tot),
        "e_states": np.asarray(mc.e_states, dtype=np.float64),
        "e_tot": float(mc.e_tot),
        "grad_sa": np.asarray(grad_sa, dtype=np.float64),
        "grads_per_root": grads_per_root,
        "nacv_01": nacv_01,
    }


# -- tests -------------------------------------------------------------------


@pytest.mark.cuda
def test_hf_energy_matches(asuka_results, pyscf_results):
    """RHF energy should match to ~1e-8 (both use exact 4c ERIs)."""
    diff = abs(asuka_results["e_hf"] - pyscf_results["e_hf"])
    assert diff < 5.0e-8, f"HF energy diff = {diff:.2e}"


@pytest.mark.cuda
def test_casscf_state_energies_match(asuka_results, pyscf_results):
    """SA-CASSCF per-state energies should match to ~1e-8."""
    e_a = asuka_results["e_states"]
    e_p = pyscf_results["e_states"]
    diff = np.max(np.abs(e_a - e_p))
    assert diff < 1.0e-7, f"CASSCF state energy max diff = {diff:.2e}"


@pytest.mark.cuda
def test_sa_gradient_matches(asuka_results, pyscf_results):
    """SA-CASSCF gradient should match to ~1e-7 (exact ERIs both sides)."""
    g_a = asuka_results["grad_sa"]
    g_p = pyscf_results["grad_sa"]
    diff = np.max(np.abs(g_a - g_p))
    assert diff < 1.0e-6, f"SA gradient max diff = {diff:.2e}"


@pytest.mark.cuda
def test_per_root_gradient_matches(asuka_results, pyscf_results):
    """Per-root SA-CASSCF gradients should match to ~1e-4.

    Looser than SA gradient because PySCF's per-root scanner re-runs CASSCF
    for each state, while ASUKA uses the analytic Z-vector approach on the
    SA solution directly.
    """
    g_a = asuka_results["grads_per_root"]
    g_p = pyscf_results["grads_per_root"]
    diff = np.max(np.abs(g_a - g_p))
    assert diff < 5.0e-4, f"Per-root gradient max diff = {diff:.2e}"


@pytest.mark.cuda
def test_nacv_matches(asuka_results, pyscf_results):
    """SA-CASSCF NACV(0,1) should match to ~1e-6 up to state-phase sign."""
    a01 = asuka_results["nacv_01"]
    p01 = pyscf_results["nacv_01"]
    diff, diff_l2 = _best_signed_diff(a01, p01)
    ref_norm = float(np.linalg.norm(p01))
    max_abs = float(np.max(np.abs(diff)))
    rel_l2 = diff_l2 / ref_norm if ref_norm > 0.0 else diff_l2
    assert max_abs < 5.0e-6, f"NACV max abs diff = {max_abs:.2e}"
    assert rel_l2 < 1.0e-4, f"NACV rel L2 = {rel_l2:.2e}"
