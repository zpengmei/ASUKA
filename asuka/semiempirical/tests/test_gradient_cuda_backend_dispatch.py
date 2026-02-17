from __future__ import annotations

import numpy as np
import pytest

from asuka.semiempirical import SemiempiricalCalculator, am1_energy_gradient
from asuka.semiempirical import gradient as gradient_mod


def _h2_case():
    symbols = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    return symbols, coords


def test_gradient_auto_cuda_fallback_to_cpu(monkeypatch):
    symbols, coords = _h2_case()
    calc = SemiempiricalCalculator(method="AM1", charge=0, device="cpu")
    scf = calc.energy(symbols, coords, coords_unit="angstrom", max_iter=120, conv_tol=1e-9)

    def _boom(*args, **kwargs):
        raise RuntimeError("forced cuda failure for fallback test")

    monkeypatch.setattr(gradient_mod, "_run_cuda_analytic", _boom)

    with pytest.warns(RuntimeWarning, match="falling back to cpu_frozen"):
        grad_auto, meta = gradient_mod.am1_gradient_from_scf(
            atomic_numbers=[1, 1],
            coords_bohr=np.asarray(coords, dtype=float) * 1.8897261254578281,
            params=calc.params,
            scf_result=scf,
            device="cuda",
            fock_mode="ri",
            gradient_backend="auto",
            return_metadata=True,
        )

    grad_cpu = gradient_mod.am1_gradient_from_scf(
        atomic_numbers=[1, 1],
        coords_bohr=np.asarray(coords, dtype=float) * 1.8897261254578281,
        params=calc.params,
        scf_result=scf,
        device="cpu",
        fock_mode="ri",
        gradient_backend="cpu_frozen",
    )

    assert np.allclose(grad_auto, grad_cpu, atol=1e-12, rtol=0.0)
    assert meta["gradient_backend_used"] == "cpu_frozen"
    assert "gradient_fallback_reason" in meta


def test_gradient_cuda_analytic_explicit_failure_surfaces(monkeypatch):
    symbols, coords = _h2_case()
    calc = SemiempiricalCalculator(method="AM1", charge=0, device="cpu")
    scf = calc.energy(symbols, coords, coords_unit="angstrom", max_iter=120, conv_tol=1e-9)

    def _boom(*args, **kwargs):
        raise RuntimeError("forced cuda failure")

    monkeypatch.setattr(gradient_mod, "_run_cuda_analytic", _boom)

    with pytest.raises(RuntimeError, match="forced cuda failure"):
        _ = gradient_mod.am1_gradient_from_scf(
            atomic_numbers=[1, 1],
            coords_bohr=np.asarray(coords, dtype=float) * 1.8897261254578281,
            params=calc.params,
            scf_result=scf,
            device="cuda",
            fock_mode="ri",
            gradient_backend="cuda_analytic",
        )


def test_energy_gradient_details_include_gradient_backend_metadata():
    symbols, coords = _h2_case()
    out = am1_energy_gradient(
        symbols,
        coords,
        coords_unit="angstrom",
        max_iter=120,
        conv_tol=1e-9,
        fock_mode="ri",
        gradient_backend="cpu_frozen",
        return_details=True,
    )
    assert out["gradient_backend_used"] == "cpu_frozen"
    assert "gradient_pack_time_s" in out
    assert "gradient_kernel_time_s" in out
    assert "gradient_post_time_s" in out
