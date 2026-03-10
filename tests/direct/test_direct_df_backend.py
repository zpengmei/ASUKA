from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


def test_rhf_direct_df_wrapper_delegates_to_streamed_df(monkeypatch):
    from asuka.hf import df_scf
    from asuka.hf import direct_scf

    calls: dict[str, object] = {}

    def _fake_rhf_df(S, hcore, B, **kwargs):
        calls["S"] = S
        calls["hcore"] = hcore
        calls["B"] = B
        calls["kwargs"] = kwargs
        return "sentinel-result"

    monkeypatch.setattr(df_scf, "rhf_df", _fake_rhf_df)

    out = direct_scf.rhf_direct_df(
        np.eye(2, dtype=np.float64),
        np.eye(2, dtype=np.float64),
        ao_basis=object(),
        aux_basis=object(),
        nelec=2,
        k_q_block=64,
        df_aux_block_naux=32,
    )

    assert out == "sentinel-result"
    assert calls["B"] is None
    kwargs = calls["kwargs"]
    assert kwargs["jk_mode"] == "streamed"
    assert kwargs["k_engine"] == "from_cocc"
    assert kwargs["k_q_block"] == 64
    assert kwargs["df_aux_block_naux"] == 32


def test_run_hf_routes_direct_df_backend(monkeypatch):
    from asuka.frontend import Molecule
    from asuka.frontend import scf as scf_frontend

    calls: dict[str, object] = {}

    @dataclass(frozen=True)
    class _DummyRun:
        two_e_backend: str | None = None
        direct_jk_ctx: object | None = None

    def _fake_run(mol, **kwargs):
        calls["mol"] = mol
        calls["kwargs"] = kwargs
        return _DummyRun()

    monkeypatch.setattr(scf_frontend, "run_rhf_direct_df", _fake_run)

    mol = Molecule.from_atoms(
        atoms=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))],
        basis="sto-3g",
    )
    out = scf_frontend.run_hf_df(
        mol,
        method="rhf",
        backend="cuda",
        two_e_backend="direct_df",
        auxbasis="autoaux",
        k_q_block=96,
    )

    assert isinstance(out, _DummyRun)
    assert out.two_e_backend == "direct_df"
    kwargs = calls["kwargs"]
    assert kwargs["auxbasis"] == "autoaux"
    assert kwargs["k_q_block"] == 96
    assert kwargs["dm0"] is None
    assert kwargs["mo_coeff0"] is None


def test_run_hf_rejects_direct_df_for_rks():
    from asuka.frontend import Molecule
    from asuka.frontend import scf as scf_frontend

    mol = Molecule.from_atoms(
        atoms=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))],
        basis="sto-3g",
    )

    with pytest.raises(NotImplementedError):
        scf_frontend.run_hf_df(
            mol,
            method="rks",
            backend="cuda",
            two_e_backend="direct_df",
        )
