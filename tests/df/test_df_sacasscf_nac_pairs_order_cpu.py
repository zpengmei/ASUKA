import numpy as np

from asuka.frontend import Molecule, run_hf
from asuka.mcscf import run_casscf
from asuka.mcscf.nac import sacasscf_nonadiabatic_couplings_df


def test_df_sacasscf_nac_pairs_are_bra_ket_cpu():
    mol = Molecule.from_atoms(
        atoms=[
            ("Li", (0.0, 0.0, 0.0)),
            ("H", (0.2, 0.1, 3.0)),
        ],
        unit="Bohr",
        basis="sto-3g",
        cart=True,
    )

    scf_out = run_hf(
        mol,
        method="rhf",
        backend="cpu",
        max_cycle=50,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )
    assert bool(getattr(scf_out.scf, "converged", False))

    mc = run_casscf(
        scf_out,
        ncore=1,
        ncas=2,
        nelecas=2,
        backend="cpu",
        max_cycle_macro=30,
        nroots=2,
        root_weights=[0.5, 0.5],
    )
    assert bool(getattr(mc, "converged", False))

    nac_all = sacasscf_nonadiabatic_couplings_df(
        scf_out,
        mc,
        pairs=None,
        mult_ediff=True,
        df_backend="cpu",
        response_term="split_orbfd",
        z_tol=1e-9,
        z_maxiter=120,
    )
    nac_subset = sacasscf_nonadiabatic_couplings_df(
        scf_out,
        mc,
        pairs=[(0, 1)],
        mult_ediff=True,
        df_backend="cpu",
        response_term="split_orbfd",
        z_tol=1e-9,
        z_maxiter=120,
    )

    assert float(np.linalg.norm(np.asarray(nac_all[0, 1], dtype=np.float64))) > 1.0e-10
    assert bool(np.allclose(nac_subset[0, 1], nac_all[0, 1], atol=1e-10, rtol=1e-8))
    assert bool(np.allclose(nac_subset[1, 0], 0.0, atol=1e-14, rtol=0.0))
