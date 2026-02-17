from __future__ import annotations

import numpy as np
import pytest

from asuka.semiempirical import (
    SemiempiricalCalculator,
    am1_energy,
    am1_energy_gradient,
    am1_gradient,
)
from asuka.semiempirical.gpu import has_cupy, has_cuda_device

CUDA_OK = has_cupy() and has_cuda_device()

_CASES = [
    ("h2", ["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)),
    (
        "hcn",
        ["H", "C", "N"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.06], [0.0, 0.0, 2.22]], dtype=float),
    ),
    (
        "ch4",
        ["C", "H", "H", "H", "H"],
        np.array(
            [
                [0.0000, 0.0000, 0.0000],
                [0.6291, 0.6291, 0.6291],
                [0.6291, -0.6291, -0.6291],
                [-0.6291, 0.6291, -0.6291],
                [-0.6291, -0.6291, 0.6291],
            ],
            dtype=float,
        ),
    ),
    (
        "h2o",
        ["O", "H", "H"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]], dtype=float),
    ),
    ("co", ["C", "O"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]], dtype=float)),
    ("n2", ["N", "N"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.10]], dtype=float)),
]


@pytest.mark.parametrize("case_name,symbols,coords_ang", _CASES)
def test_am1_gradient_shape_and_finite(case_name, symbols, coords_ang):
    del case_name
    grad = am1_gradient(
        symbols,
        coords_ang,
        coords_unit="angstrom",
        max_iter=120,
        conv_tol=1e-9,
        fock_mode="ri",
    )
    assert grad.shape == (len(symbols), 3)
    assert np.all(np.isfinite(grad))
    assert np.max(np.abs(np.sum(grad, axis=0))) <= 1e-6


def test_am1_energy_gradient_matches_energy_api():
    symbols = ["H", "H"]
    coords_ang = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    e_ref = am1_energy(
        symbols,
        coords_ang,
        coords_unit="angstrom",
        max_iter=120,
        conv_tol=1e-9,
        fock_mode="ri",
    )
    e, grad = am1_energy_gradient(
        symbols,
        coords_ang,
        coords_unit="angstrom",
        max_iter=120,
        conv_tol=1e-9,
        fock_mode="ri",
    )
    assert abs(float(e) - float(e_ref)) <= 1e-10
    assert grad.shape == (2, 3)
    assert np.all(np.isfinite(grad))


def test_am1_energy_gradient_return_details_includes_gradient():
    symbols = ["H", "H"]
    coords_ang = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    out = am1_energy_gradient(
        symbols,
        coords_ang,
        coords_unit="angstrom",
        max_iter=120,
        conv_tol=1e-9,
        fock_mode="ri",
        return_details=True,
    )
    assert "energy_total" in out
    assert "gradient" in out
    grad = np.asarray(out["gradient"], dtype=float)
    assert grad.shape == (2, 3)
    assert np.all(np.isfinite(grad))


def test_calculator_gradient_methods():
    calc = SemiempiricalCalculator(method="AM1", charge=0, device="cpu")
    symbols = ["H", "H"]
    coords_ang = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    scf_res, grad = calc.energy_gradient(
        symbols,
        coords_ang,
        coords_unit="angstrom",
        max_iter=120,
        conv_tol=1e-9,
        fock_mode="ri",
    )
    assert scf_res.converged
    assert grad.shape == (2, 3)
    grad2 = calc.gradient(
        symbols,
        coords_ang,
        coords_unit="angstrom",
        max_iter=120,
        conv_tol=1e-9,
        fock_mode="ri",
    )
    assert np.allclose(grad, grad2, atol=1e-12, rtol=0.0)


def test_gradient_rejects_invalid_fock_mode():
    symbols = ["H", "H"]
    coords_ang = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    with pytest.raises(ValueError):
        _ = am1_gradient(
            symbols,
            coords_ang,
            coords_unit="angstrom",
            max_iter=80,
            fock_mode="bad-mode",
        )


def test_gradient_rejects_invalid_gradient_backend():
    symbols = ["H", "H"]
    coords_ang = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    with pytest.raises(ValueError, match="gradient_backend"):
        _ = am1_gradient(
            symbols,
            coords_ang,
            coords_unit="angstrom",
            max_iter=80,
            fock_mode="ri",
            gradient_backend="bad-backend",
        )


def test_gradient_cuda_backend_requires_cuda_device_selection():
    symbols = ["H", "H"]
    coords_ang = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    with pytest.raises(ValueError, match="requires device='cuda'"):
        _ = am1_gradient(
            symbols,
            coords_ang,
            coords_unit="angstrom",
            device="cpu",
            max_iter=80,
            fock_mode="ri",
            gradient_backend="cuda_analytic",
        )


def test_gradient_reports_unconverged_scf():
    symbols = ["H", "H"]
    coords_ang = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    with pytest.raises(RuntimeError, match="converged SCF"):
        _ = am1_gradient(
            symbols,
            coords_ang,
            coords_unit="angstrom",
            max_iter=1,
            conv_tol=1e-14,
            fock_mode="ri",
        )


def test_gradient_reports_unsupported_element():
    symbols = ["F", "H"]
    coords_ang = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.92]], dtype=float)
    with pytest.raises(ValueError, match="No parameters for element"):
        _ = am1_gradient(symbols, coords_ang, coords_unit="angstrom", max_iter=80, fock_mode="ri")


@pytest.mark.cuda
@pytest.mark.skipif(not CUDA_OK, reason="CuPy/CUDA unavailable")
def test_am1_gradient_cuda_smoke():
    symbols = ["H", "H"]
    coords_ang = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    grad = am1_gradient(
        symbols,
        coords_ang,
        coords_unit="angstrom",
        device="cuda",
        max_iter=120,
        conv_tol=1e-9,
        fock_mode="w",
    )
    assert grad.shape == (2, 3)
    assert np.all(np.isfinite(grad))
