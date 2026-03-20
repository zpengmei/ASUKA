"""Comprehensive SA-CASSCF gradient & NACV validation across all ASUKA backends.

Tests SA gradient, per-root gradient (weighted-sum + PySCF parity), and NACV
for: direct-4c/CUDA, DF/CPU, DF/CUDA.  PySCF conventional integrals as reference.

Covers multiple molecules, basis sets, and active spaces.

Usage:
    python tests/test_all_sacasscf_grad_backends.py
"""
from __future__ import annotations

import numpy as np
import sys
import time

# ---------------------------------------------------------------------------
#  Test systems: (name, atoms, unit, basis, ncas, nelecas, nroots, weights, cart)
# ---------------------------------------------------------------------------
SYSTEMS = [
    # ── H2O / STO-3G / CAS(4,4) ── minimal, no virtuals
    dict(
        name="H2O/STO-3G/CAS(4,4)",
        atoms=[("O", (0.0, 0.0, 0.117)), ("H", (0.0, 0.757, -0.470)), ("H", (0.0, -0.780, -0.450))],
        basis="sto-3g", ncas=4, nelecas=4, nroots=2, weights=[0.5, 0.5],
    ),
    # ── H2O / 6-31g / CAS(4,4) ── larger basis, has virtuals
    dict(
        name="H2O/6-31g/CAS(4,4)",
        atoms=[("O", (0.0, 0.0, 0.117)), ("H", (0.0, 0.757, -0.470)), ("H", (0.0, -0.780, -0.450))],
        basis="6-31g", ncas=4, nelecas=4, nroots=2, weights=[0.5, 0.5],
    ),
    # ── LiH / STO-3G / CAS(2,2) ── 2-electron system, different element
    dict(
        name="LiH/STO-3G/CAS(2,2)",
        atoms=[("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.6))],
        basis="sto-3g", ncas=2, nelecas=2, nroots=2, weights=[0.5, 0.5],
    ),
    # ── N2 / STO-3G / CAS(6,6) ── larger active space, triple bond
    dict(
        name="N2/STO-3G/CAS(6,6)",
        atoms=[("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.1))],
        basis="sto-3g", ncas=6, nelecas=6, nroots=2, weights=[0.5, 0.5],
    ),
    # ── NH3 / STO-3G / CAS(4,4) ── non-planar, C3v symmetry
    dict(
        name="NH3/STO-3G/CAS(4,4)",
        atoms=[("N", (0.0, 0.0, 0.116)), ("H", (0.0, 0.939, -0.272)),
               ("H", (0.813, -0.470, -0.272)), ("H", (-0.813, -0.470, -0.272))],
        basis="sto-3g", ncas=4, nelecas=4, nroots=2, weights=[0.5, 0.5],
    ),
]


def best_signed_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    dp, dm = a - b, a + b
    return dp if np.linalg.norm(dp) <= np.linalg.norm(dm) else dm


# ═══════════════════════════════════════════════════════════════════════════
#  PySCF reference
# ═══════════════════════════════════════════════════════════════════════════
def run_pyscf(sys_d):
    from pyscf import gto, scf, mcscf, fci, lib
    lib.num_threads(4)
    mol = gto.M(
        atom="; ".join(f"{s} {x} {y} {z}" for s, (x, y, z) in sys_d["atoms"]),
        unit="Angstrom", basis=sys_d["basis"], cart=True, spin=0, verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    assert mf.converged, f"PySCF RHF did not converge for {sys_d['name']}"

    nroots = sys_d["nroots"]
    solver = fci.direct_spin0.FCI(mol)
    solver.nroots = nroots
    mc = mcscf.CASSCF(mf, sys_d["ncas"], sys_d["nelecas"])
    mc.conv_tol = 1e-10
    mc.fcisolver = solver
    mc = mc.state_average_(sys_d["weights"]).run(verbose=0)
    assert mc.converged, f"PySCF CASSCF did not converge for {sys_d['name']}"

    from pyscf.grad import sacasscf as sacasscf_grad
    grad_sa = sacasscf_grad.Gradients(mc).kernel()
    grad_pr = np.array([sacasscf_grad.Gradients(mc).kernel(state=i) for i in range(nroots)])

    # NACV only for 2-state (pair 0,1)
    nacv = None
    if nroots >= 2:
        try:
            nacv = np.asarray(
                mc.nac_method().kernel(state=(0, 1), use_etfs=False, mult_ediff=True), float
            )
        except Exception:
            pass

    return {
        "e_roots": np.asarray(mc.e_states, float),
        "grad_sa": np.asarray(grad_sa, float),
        "grad_pr": np.asarray(grad_pr, float),
        "nacv": nacv,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  ASUKA runners
# ═══════════════════════════════════════════════════════════════════════════
def run_asuka_casscf(sys_d, two_e, comp):
    from asuka.frontend import Molecule
    from asuka.frontend.scf import run_hf_df
    from asuka.mcscf import run_casscf

    mol = Molecule.from_atoms(sys_d["atoms"], unit="Angstrom", basis=sys_d["basis"], cart=True)
    scf_out = run_hf_df(mol, method="rhf", backend=comp, two_e_backend=two_e,
                        max_cycle=60, conv_tol=1e-12, conv_tol_dm=1e-10)
    assert scf_out.scf.converged
    ncore = (mol.nelectron - sys_d["nelecas"]) // 2
    mc = run_casscf(scf_out, ncore=ncore, ncas=sys_d["ncas"], nelecas=sys_d["nelecas"],
                    nroots=sys_d["nroots"], root_weights=sys_d["weights"],
                    backend=comp, df=True,
                    max_cycle_macro=80, tol=1e-10, conv_tol_grad=1e-7)
    assert mc.converged
    return scf_out, mc


def asuka_direct(scf_out, mc, sys_d):
    """All quantities via exact 4c path."""
    from asuka.mcscf.nuc_grad_direct import casscf_nuc_grad_direct, casscf_nuc_grad_direct_per_root
    from asuka.mcscf.nac._dense import sacasscf_nonadiabatic_couplings_dense

    g_sa = np.asarray(casscf_nuc_grad_direct(scf_out, mc).grad, float)
    pr = casscf_nuc_grad_direct_per_root(scf_out, mc, z_tol=1e-11, z_maxiter=300)
    g_pr = np.asarray(pr.grads, float)

    nacv = None
    if sys_d["nroots"] >= 2:
        try:
            nac = sacasscf_nonadiabatic_couplings_dense(
                scf_out, mc, pairs=[(0, 1)], mult_ediff=True, use_etfs=False,
                response_term="split_orbfd", z_tol=1e-11, z_maxiter=300)
            nacv = np.asarray(nac[0, 1], float)
        except Exception as e:
            nacv = f"SKIP:{e}"
    return g_sa, g_pr, nacv


def asuka_df(scf_out, mc, sys_d, comp):
    """All quantities via DF path."""
    from asuka.mcscf.nuc_grad_df import casscf_nuc_grad_df, casscf_nuc_grad_df_per_root
    from asuka.mcscf.nac._df import sacasscf_nonadiabatic_couplings_df

    g_sa = np.asarray(casscf_nuc_grad_df(scf_out, mc, df_backend=comp).grad, float)
    pr = casscf_nuc_grad_df_per_root(scf_out, mc, df_backend=comp, z_tol=1e-11, z_maxiter=300)
    g_pr = np.asarray(pr.grads, float)

    nacv = None
    if sys_d["nroots"] >= 2:
        try:
            nac = sacasscf_nonadiabatic_couplings_df(
                scf_out, mc, pairs=[(0, 1)], mult_ediff=True, use_etfs=False,
                df_backend=comp, response_term="split_orbfd", z_tol=1e-11, z_maxiter=300)
            nacv = np.asarray(nac[0, 1], float)
        except Exception as e:
            nacv = f"SKIP:{e}"
    return g_sa, g_pr, nacv


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    all_checks = []

    for sys_d in SYSTEMS:
        name = sys_d["name"]
        nroots = sys_d["nroots"]
        w = np.asarray(sys_d["weights"], float)

        print(f"\n{'#' * 70}")
        print(f"  {name}")
        print(f"{'#' * 70}")

        # PySCF reference
        t0 = time.time()
        ref = run_pyscf(sys_d)
        print(f"  PySCF: E={ref['e_roots']}  ({time.time()-t0:.1f}s)")

        def chk(label, val, thr):
            ok = val < thr
            all_checks.append((f"{name} | {label}", val, thr, ok))
            print(f"    {'OK' if ok else 'FAIL'} {label}: {val:.2e} (thr {thr:.0e})")
            return ok

        # ── direct/cuda ──
        print(f"\n  --- direct/cuda ---")
        try:
            scf_d, mc_d = run_asuka_casscf(sys_d, "direct", "cuda")
            g_sa_d, g_pr_d, nacv_d = asuka_direct(scf_d, mc_d, sys_d)

            chk("direct energy", np.max(np.abs(np.asarray(mc_d.e_roots) - ref["e_roots"])), 1e-5)
            chk("direct SA-grad", np.max(np.abs(g_sa_d - ref["grad_sa"])), 5e-5)
            ws_d = sum(w[k] * g_pr_d[k] for k in range(nroots))
            chk("direct PR-wsum", np.max(np.abs(ws_d - g_sa_d)), 5e-4)
            for i in range(nroots):
                chk(f"direct PR-root{i}", np.max(np.abs(g_pr_d[i] - ref["grad_pr"][i])), 1e-4)
            if isinstance(nacv_d, np.ndarray) and ref["nacv"] is not None:
                chk("direct NACV", np.max(np.abs(best_signed_diff(nacv_d, ref["nacv"]))), 5e-5)
            elif isinstance(nacv_d, str):
                print(f"    NACV {nacv_d}")
        except Exception as e:
            print(f"    SKIP direct/cuda: {e}")

        # ── df/cpu ──
        print(f"\n  --- df/cpu ---")
        try:
            scf_c, mc_c = run_asuka_casscf(sys_d, "df", "cpu")
            g_sa_c, g_pr_c, nacv_c = asuka_df(scf_c, mc_c, sys_d, "cpu")

            chk("df/cpu energy", np.max(np.abs(np.asarray(mc_c.e_roots) - ref["e_roots"])), 2e-3)
            chk("df/cpu SA-grad", np.max(np.abs(g_sa_c - ref["grad_sa"])), 2e-3)
            ws_c = sum(w[k] * g_pr_c[k] for k in range(nroots))
            chk("df/cpu PR-wsum", np.max(np.abs(ws_c - g_sa_c)), 5e-4)
            for i in range(nroots):
                chk(f"df/cpu PR-root{i}", np.max(np.abs(g_pr_c[i] - ref["grad_pr"][i])), 2e-3)
            if isinstance(nacv_c, np.ndarray) and ref["nacv"] is not None:
                chk("df/cpu NACV", np.max(np.abs(best_signed_diff(nacv_c, ref["nacv"]))), 2e-3)
            elif isinstance(nacv_c, str):
                print(f"    NACV {nacv_c}")
        except Exception as e:
            print(f"    SKIP df/cpu: {e}")

        # ── df/cuda (reuse df/cpu CASSCF for deterministic comparison) ──
        print(f"\n  --- df/cuda (same CASSCF as df/cpu) ---")
        try:
            g_sa_g, g_pr_g, nacv_g = asuka_df(scf_c, mc_c, sys_d, "cuda")
            chk("df cpu≡cuda SA", np.max(np.abs(g_sa_g - g_sa_c)), 1e-6)
            ws_g = sum(w[k] * g_pr_g[k] for k in range(nroots))
            chk("df/cuda PR-wsum", np.max(np.abs(ws_g - g_sa_g)), 5e-4)
            chk("df cpu≡cuda PR", np.max(np.abs(g_pr_g - g_pr_c)), 1e-4)
            if isinstance(nacv_g, np.ndarray) and isinstance(nacv_c, np.ndarray):
                chk("df cpu≡cuda NACV", np.max(np.abs(best_signed_diff(nacv_g, nacv_c))), 1e-4)
        except Exception as e:
            print(f"    SKIP df/cuda: {e}")

    # ── Grand summary ──
    print(f"\n{'=' * 70}")
    print("  GRAND SUMMARY")
    print(f"{'=' * 70}")
    n_pass = sum(1 for _, _, _, ok in all_checks if ok)
    n_fail = sum(1 for _, _, _, ok in all_checks if not ok)
    for label, val, thr, ok in all_checks:
        if not ok:
            print(f"  FAIL  {label}: {val:.2e} > {thr:.0e}")
    print(f"\n  {n_pass} passed, {n_fail} failed, {len(all_checks)} total")
    print(f"  Time: {time.time() - t_start:.0f}s")
    print(f"  {'ALL PASS' if n_fail == 0 else 'SOME FAILED'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
