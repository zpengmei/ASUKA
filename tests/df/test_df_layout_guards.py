from types import SimpleNamespace

import numpy as np
import pytest

from asuka.mcscf.nuc_grad_df import (
    casscf_nuc_grad_df,
    casscf_nuc_grad_df_per_root,
    casci_nuc_grad_df_relaxed,
    casci_nuc_grad_df_unrelaxed,
)


def _fake_scf_out_qmn() -> SimpleNamespace:
    mol = SimpleNamespace(cart=True)
    int1e = SimpleNamespace(S=np.eye(4, dtype=np.float64))
    # Qmn-like layout: first two dims are not (nao, nao)
    df_B = np.zeros((9, 4, 4), dtype=np.float64)
    return SimpleNamespace(mol=mol, int1e=int1e, df_B=df_B)


def test_casscf_nuc_grad_df_rejects_non_mnq_df_layout():
    with pytest.raises(ValueError, match="df_layout='mnQ'"):
        casscf_nuc_grad_df(_fake_scf_out_qmn(), SimpleNamespace())


def test_casscf_nuc_grad_df_per_root_rejects_non_mnq_df_layout():
    with pytest.raises(ValueError, match="df_layout='mnQ'"):
        casscf_nuc_grad_df_per_root(_fake_scf_out_qmn(), SimpleNamespace())


def test_casci_unrelaxed_rejects_non_mnq_df_layout():
    with pytest.raises(ValueError, match="df_layout='mnQ'"):
        casci_nuc_grad_df_unrelaxed(_fake_scf_out_qmn(), SimpleNamespace())


def test_casci_relaxed_rejects_non_mnq_df_layout():
    with pytest.raises(ValueError, match="df_layout='mnQ'"):
        casci_nuc_grad_df_relaxed(_fake_scf_out_qmn(), SimpleNamespace())
