from __future__ import annotations

import numpy as np

from asuka.mrci import ic_rdm


def test_rdm2_extint_extext_wrapper_delegates(monkeypatch):
    captured = {}

    def _stub(ic_res, **kwargs):
        captured.update(kwargs)
        return np.zeros((1, 1, 1, 1), dtype=np.float64)

    monkeypatch.setattr(ic_rdm, "_build_rdm2_extint_extext_block_impl", _stub)
    out = ic_rdm.ic_mrcisd_make_rdm2_extint_extext_phase3(object(), ci_cas=np.asarray([1.0], dtype=np.float64))

    assert out.shape == (1, 1, 1, 1)
    assert captured["make_rdm12_fn"] is ic_rdm.ic_mrcisd_make_rdm12
    assert captured["pair_to_mat_builder"] is ic_rdm._fic_build_doubles_pair_to_mat


def test_rdm2_intext_extext_wrapper_delegates(monkeypatch):
    captured = {}

    def _stub(ic_res, **kwargs):
        captured.update(kwargs)
        return np.zeros((1, 1, 1, 1), dtype=np.float64)

    monkeypatch.setattr(ic_rdm, "_build_rdm2_intext_extext_block_impl", _stub)
    out = ic_rdm.ic_mrcisd_make_rdm2_intext_extext_phase3(object(), ci_cas=np.asarray([1.0], dtype=np.float64))

    assert out.shape == (1, 1, 1, 1)
    assert captured["make_rdm12_fn"] is ic_rdm.ic_mrcisd_make_rdm12
    assert captured["pair_to_mat_builder"] is ic_rdm._fic_build_doubles_pair_to_mat
