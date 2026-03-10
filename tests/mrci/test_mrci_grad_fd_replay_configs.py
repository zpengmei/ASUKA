from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from asuka.frontend.scf import DFRunConfig
from asuka.mcscf.casscf import CASSCFRunConfig
from asuka.mrci.grad_fd import _run_casscf_like_reference, _run_hf_like_reference


def test_run_hf_like_reference_prefers_persisted_df_run_config(monkeypatch):
    calls: list[dict] = []

    def _fake_run_hf_df(mol, **kwargs):
        calls.append({"mol": mol, **kwargs})
        return "scf-sentinel"

    monkeypatch.setattr("asuka.frontend.scf.run_hf_df", _fake_run_hf_df)

    scf_out0 = SimpleNamespace(
        df_B=np.zeros((1, 1)),
        thc_factors=None,
        auxbasis_name="wrong-auxbasis",
        scf=SimpleNamespace(method="RHF", mo_coeff=np.eye(2)),
        df_run_config=DFRunConfig(
            hf_method="rhf",
            basis="sto-3g",
            auxbasis={"Li": "autoaux"},
            df_config={"screen": 1.0e-12},
            expand_contractions=False,
            backend="cuda",
            max_cycle=23,
            conv_tol=1.0e-10,
            conv_tol_dm=1.0e-8,
            diis=True,
            diis_start_cycle=2,
            diis_space=9,
            damping=0.1,
            level_shift=0.2,
            k_q_block=77,
            cublas_math_mode="tf32",
            init_fock_cycles=3,
        ),
    )

    mol = object()
    out = _run_hf_like_reference(scf_out0, mol)

    assert out == "scf-sentinel"
    assert len(calls) == 1
    call = calls[0]
    assert call["mol"] is mol
    assert call["method"] == "rhf"
    assert call["backend"] == "cuda"
    assert call["basis"] == "sto-3g"
    assert call["auxbasis"] == {"Li": "autoaux"}
    assert call["df_config"] == {"screen": 1.0e-12}
    assert call["expand_contractions"] is False
    assert call["k_q_block"] == 77
    assert call["cublas_math_mode"] == "tf32"
    assert call["init_fock_cycles"] == 3
    assert np.allclose(call["mo_coeff0"], np.eye(2))
    assert call["guess"] is scf_out0


def test_run_casscf_like_reference_replays_persisted_run_config(monkeypatch):
    calls: list[dict] = []

    def _fake_run_casscf(scf_out, **kwargs):
        calls.append({"scf_out": scf_out, **kwargs})
        return "casscf-sentinel"

    monkeypatch.setattr("asuka.mcscf.casscf.run_casscf", _fake_run_casscf)

    ref0 = SimpleNamespace(
        ncore=1,
        ncas=2,
        nelecas=2,
        nroots=2,
        root_weights=np.asarray([0.25, 0.75], dtype=np.float64),
        run_config=CASSCFRunConfig(
            backend="cuda",
            df=True,
            matvec_backend="cuda_eri_mat",
            nroots=2,
            root_weights=(0.25, 0.75),
            kwargs={
                "orbital_optimizer": "lbfgs",
                "max_cycle_macro": 17,
                "tol": 1.0e-9,
            },
        ),
    )

    scf_out = object()
    out = _run_casscf_like_reference(scf_out, ref0)

    assert out == "casscf-sentinel"
    assert len(calls) == 1
    call = calls[0]
    assert call["scf_out"] is scf_out
    assert call["ncore"] == 1
    assert call["ncas"] == 2
    assert call["nelecas"] == 2
    assert call["backend"] == "cuda"
    assert call["df"] is True
    assert call["guess"] is ref0
    assert call["matvec_backend"] == "cuda_eri_mat"
    assert call["nroots"] == 2
    assert call["root_weights"] == [0.25, 0.75]
    assert call["orbital_optimizer"] == "lbfgs"
    assert call["max_cycle_macro"] == 17
    assert call["tol"] == 1.0e-9
