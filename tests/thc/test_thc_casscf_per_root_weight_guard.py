from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_thc_casscf_per_root_rejects_unequal_sa_weights_early():
    from asuka.mcscf.nuc_grad_thc import casscf_nuc_grad_thc_per_root

    fake_mc = SimpleNamespace(
        nroots=2,
        root_weights=[0.7, 0.3],
    )

    with pytest.raises(NotImplementedError, match="equal SA weights"):
        casscf_nuc_grad_thc_per_root(object(), fake_mc)
