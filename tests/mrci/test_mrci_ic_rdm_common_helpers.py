from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from asuka.mrci import ic_rdm
from asuka.mrci.ic_rdm_common import infer_n_act_n_virt, require_internal_external_contiguous


def test_infer_n_act_n_virt_parses_spaces():
    ic_res = SimpleNamespace(spaces=SimpleNamespace(n_internal=4, n_external=6))
    assert infer_n_act_n_virt(ic_res) == (4, 6)


def test_require_internal_external_contiguous_guards_shape_and_order():
    spaces_ok = SimpleNamespace(internal=np.array([0, 1, 2]), external=np.array([3, 4]))
    require_internal_external_contiguous(spaces_ok, n_act=3, n_virt=2)

    spaces_bad = SimpleNamespace(internal=np.array([0, 2, 1]), external=np.array([3, 4]))
    with pytest.raises(NotImplementedError, match="contiguous correlated ordering"):
        require_internal_external_contiguous(spaces_bad, n_act=3, n_virt=2)


def test_ic_rdm_wrapper_delegates_infer(monkeypatch):
    called = {}

    def _stub(ic_res):
        called["ok"] = True
        return (2, 3)

    monkeypatch.setattr(ic_rdm, "_infer_n_act_n_virt_common", _stub)
    assert ic_rdm._infer_n_act_n_virt(object()) == (2, 3)
    assert called == {"ok": True}


def test_ic_rdm_wrapper_delegates_contiguous_guard(monkeypatch):
    called = {}

    def _stub(spaces, *, n_act, n_virt):
        called["args"] = (spaces, n_act, n_virt)

    sentinel = object()
    monkeypatch.setattr(ic_rdm, "_require_internal_external_contiguous_common", _stub)
    ic_rdm._require_internal_external_contiguous(sentinel, n_act=1, n_virt=2)
    assert called["args"] == (sentinel, 1, 2)
