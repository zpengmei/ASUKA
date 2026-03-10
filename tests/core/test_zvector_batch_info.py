import numpy as np

import asuka.mcscf.zvector as zvec


def _fake_result(*, solver: str, backend: str, tag: int) -> zvec.MCSCFZVectorResult:
    z = np.asarray([float(tag)], dtype=np.float64)
    info = {"solver": str(solver), "backend": str(backend), "matvec_calls": 3, "niter": 2}
    return zvec.MCSCFZVectorResult(
        converged=True,
        niter=2,
        residual_norm=0.0,
        z_orb=z,
        z_ci=[],
        z_packed=z,
        info=info,
    )


def test_zvector_batch_info_reports_cuda_backend(monkeypatch):
    seq = [
        _fake_result(solver="gcrotmk_gpu", backend="cuda", tag=0),
        _fake_result(solver="gcrotmk_gpu", backend="cuda", tag=1),
    ]

    state = {"i": 0}

    def _fake_solve_mcscf_zvector(*args, **kwargs):
        idx = state["i"]
        state["i"] += 1
        return seq[idx]

    monkeypatch.setattr(zvec, "solve_mcscf_zvector", _fake_solve_mcscf_zvector)

    out = zvec.solve_mcscf_zvector_batch(
        object(),
        rhs_orb_list=[np.ones(3), np.ones(3)],
        rhs_ci_list=[None, None],
        method="gcrotmk",
    )
    assert out.info["solver"] == "gcrotmk"
    assert out.info["backend"] == "cuda"
    assert out.info["solver_detail"] == ["gcrotmk_gpu"]
    assert out.info["backend_detail"] == ["cuda"]


def test_zvector_batch_info_reports_mixed_backend(monkeypatch):
    seq = [
        _fake_result(solver="gcrotmk_gpu", backend="cuda", tag=0),
        _fake_result(solver="gcrotmk", backend="cpu", tag=1),
    ]

    state = {"i": 0}

    def _fake_solve_mcscf_zvector(*args, **kwargs):
        idx = state["i"]
        state["i"] += 1
        return seq[idx]

    monkeypatch.setattr(zvec, "solve_mcscf_zvector", _fake_solve_mcscf_zvector)

    out = zvec.solve_mcscf_zvector_batch(
        object(),
        rhs_orb_list=[np.ones(4), np.ones(2)],
        rhs_ci_list=[None, None],
        method="gcrotmk",
    )
    assert out.info["backend"] == "mixed"
    assert out.info["solver_detail"] == ["gcrotmk", "gcrotmk_gpu"]
    assert out.info["backend_detail"] == ["cpu", "cuda"]
