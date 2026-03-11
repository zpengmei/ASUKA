from __future__ import annotations

import numpy as np

from asuka.mrci import ic_rdm


def test_rdm1_internal_block_wrapper_delegates(monkeypatch):
    captured = {}

    def _stub(ic_res, **kwargs):
        captured.update(kwargs)
        return np.asarray([[1.0]], dtype=np.float64)

    monkeypatch.setattr(ic_rdm, "_build_rdm1_internal_block_impl", _stub)

    out = ic_rdm.ic_mrcisd_make_rdm1_internal_phase3(object(), ci_cas=np.asarray([1.0], dtype=np.float64))
    np.testing.assert_allclose(out, np.asarray([[1.0]], dtype=np.float64))
    assert captured["make_rdm12_fn"] is ic_rdm.ic_mrcisd_make_rdm12


def test_rdm1_external_internal_block_wrapper_delegates(monkeypatch):
    captured = {}

    def _stub(ic_res, **kwargs):
        captured.update(kwargs)
        return np.asarray([[2.0, 3.0]], dtype=np.float64)

    monkeypatch.setattr(ic_rdm, "_build_rdm1_external_internal_block_impl", _stub)

    out = ic_rdm.ic_mrcisd_make_rdm1_ext_int_phase3(object(), ci_cas=np.asarray([1.0], dtype=np.float64))
    np.testing.assert_allclose(out, np.asarray([[2.0, 3.0]], dtype=np.float64))
    assert captured["make_rdm12_fn"] is ic_rdm.ic_mrcisd_make_rdm12


def test_rdm1_external_external_block_wrapper_delegates(monkeypatch):
    captured = {}

    def _stub(ic_res, **kwargs):
        captured.update(kwargs)
        return np.asarray([[4.0]], dtype=np.float64)

    monkeypatch.setattr(ic_rdm, "_build_rdm1_external_external_block_impl", _stub)

    out = ic_rdm.ic_mrcisd_make_rdm1_ext_ext_phase3(object(), ci_cas=np.asarray([1.0], dtype=np.float64))
    np.testing.assert_allclose(out, np.asarray([[4.0]], dtype=np.float64))
    assert captured["make_rdm12_fn"] is ic_rdm.ic_mrcisd_make_rdm12


def test_rdm1_block_assembler_assembles_blocks(monkeypatch):
    monkeypatch.setattr(ic_rdm, "_infer_n_act_n_virt", lambda _ic_res: (2, 1))
    monkeypatch.setattr(
        ic_rdm,
        "ic_mrcisd_make_rdm1_internal_phase3",
        lambda *args, **kwargs: np.asarray([[10.0, 11.0], [12.0, 13.0]], dtype=np.float64),
    )
    monkeypatch.setattr(
        ic_rdm,
        "ic_mrcisd_make_rdm1_ext_int_phase3",
        lambda *args, **kwargs: np.asarray([[20.0, 21.0]], dtype=np.float64),
    )
    monkeypatch.setattr(
        ic_rdm,
        "ic_mrcisd_make_rdm1_ext_ext_phase3",
        lambda *args, **kwargs: np.asarray([[30.0]], dtype=np.float64),
    )

    out = ic_rdm.ic_mrcisd_make_rdm1_phase3(object(), ci_cas=np.asarray([1.0], dtype=np.float64), backend="direct")
    expect = np.asarray(
        [
            [10.0, 11.0, 20.0],
            [12.0, 13.0, 21.0],
            [20.0, 21.0, 30.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out, expect)
