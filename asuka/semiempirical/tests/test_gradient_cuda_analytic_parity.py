from __future__ import annotations

import numpy as np
import pytest

from asuka.semiempirical import am1_gradient
from asuka.semiempirical.gpu import has_cupy, has_cuda_device

CUDA_OK = has_cupy() and has_cuda_device()


def _water_chain_16_case():
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.586],
            [0.0, -0.757, 0.586],
        ],
        dtype=float,
    )
    coords = []
    symbols = []
    for i in range(16):
        shift = np.array([(i % 8) * 3.0, ((i // 8) % 8) * 3.0, (i // 64) * 3.0], dtype=float)
        symbols.extend(["O", "H", "H"])
        coords.extend((base + shift).tolist())
    return symbols, np.asarray(coords, dtype=float)


_SMALL_CASES = [
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


@pytest.mark.cuda
@pytest.mark.skipif(not CUDA_OK, reason="CuPy/CUDA unavailable")
@pytest.mark.parametrize("case_name,symbols,coords_ang", _SMALL_CASES)
@pytest.mark.parametrize("fock_mode", ["ri", "w"])
def test_am1_gradient_cuda_analytic_parity_small(case_name, symbols, coords_ang, fock_mode):
    del case_name
    grad_cpu = am1_gradient(
        symbols,
        coords_ang,
        coords_unit="angstrom",
        device="cpu",
        max_iter=120,
        conv_tol=1e-9,
        fock_mode=fock_mode,
        gradient_backend="cpu_frozen",
    )
    grad_cuda = am1_gradient(
        symbols,
        coords_ang,
        coords_unit="angstrom",
        device="cuda",
        max_iter=120,
        conv_tol=1e-9,
        fock_mode=fock_mode,
        gradient_backend="cuda_analytic",
    )

    d = np.asarray(grad_cuda) - np.asarray(grad_cpu)
    assert float(np.max(np.abs(d))) <= 1e-5
    assert float(np.max(np.abs(np.sum(grad_cuda, axis=0)))) <= 1e-6


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not CUDA_OK, reason="CuPy/CUDA unavailable")
def test_am1_gradient_cuda_analytic_parity_water16():
    symbols, coords_ang = _water_chain_16_case()
    grad_cpu = am1_gradient(
        symbols,
        coords_ang,
        coords_unit="angstrom",
        device="cpu",
        max_iter=120,
        conv_tol=1e-9,
        fock_mode="w",
        gradient_backend="cpu_frozen",
    )
    grad_cuda = am1_gradient(
        symbols,
        coords_ang,
        coords_unit="angstrom",
        device="cuda",
        max_iter=120,
        conv_tol=1e-9,
        fock_mode="w",
        gradient_backend="cuda_analytic",
    )
    d = np.asarray(grad_cuda) - np.asarray(grad_cpu)
    assert float(np.max(np.abs(d))) <= 1e-5
    assert float(np.max(np.abs(np.sum(grad_cuda, axis=0)))) <= 1e-6
