from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.cuda
def test_df_casscf_per_root_grad_matches_fd_component():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.zeros((1,), dtype=cp.float64)
        _ = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        pytest.skip("CuPy is present but a CUDA device is unavailable")

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
            two_e_backend="df",
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
            df=True,
            nroots=2,
            root_weights=[0.5, 0.5],
            max_cycle_macro=40,
            tol=1e-9,
            conv_tol_grad=1e-7,
        )
        assert bool(getattr(mc, "converged", False))
        out = casscf_nuc_grad_per_root(
            scf_out,
            mc,
            backend="df",
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

    assert max(errs) < 1.0e-5, f"DF per-root FD mismatch too large: max_err={max(errs):.3e}"

