from __future__ import annotations

import numpy as np

from asuka.frontend import Molecule
from asuka.frontend.scf import run_hf_df


def test_run_hf_df_cpu_populates_df_metric_cholesky() -> None:
    mol = Molecule.from_atoms(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))],
        unit="Bohr",
        basis="sto-3g",
        cart=True,
    )
    scf_out = run_hf_df(mol, backend="cpu")

    assert getattr(scf_out, "df_L", None) is not None
    df_b = np.asarray(getattr(scf_out, "df_B"), dtype=np.float64)
    df_l = np.asarray(getattr(scf_out, "df_L"), dtype=np.float64)
    assert df_l.shape == (df_b.shape[2], df_b.shape[2])
