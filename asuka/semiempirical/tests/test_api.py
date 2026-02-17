from __future__ import annotations

import numpy as np
import pytest

from asuka.semiempirical import SemiempiricalCalculator, am1_energy
from asuka.semiempirical.gpu import has_cupy, has_cuda_device


def test_am1_energy_cpu_h2_runs():
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    e = am1_energy(["H", "H"], coords, coords_unit="angstrom", max_iter=80)
    assert isinstance(e, float)
    assert np.isfinite(e)


def test_am1_energy_cpu_accepts_fock_mode_kwarg():
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    e = am1_energy(
        ["H", "H"],
        coords,
        coords_unit="angstrom",
        max_iter=80,
        fock_mode="ri",
    )
    assert isinstance(e, float)
    assert np.isfinite(e)


def test_am1_energy_cpu_accepts_fock_mode_auto():
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    e = am1_energy(
        ["H", "H"],
        coords,
        coords_unit="angstrom",
        max_iter=80,
        fock_mode="auto",
    )
    assert isinstance(e, float)
    assert np.isfinite(e)


def test_am1_details_include_expected_keys():
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    out = am1_energy(
        ["H", "H"],
        coords,
        coords_unit="angstrom",
        max_iter=80,
        return_details=True,
    )
    assert out["method"] == "AM1"
    assert "energy_total" in out
    assert "eps" in out


def test_pm7_scaffold_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        SemiempiricalCalculator(method="PM7")


@pytest.mark.cuda
@pytest.mark.skipif(not has_cupy() or not has_cuda_device(), reason="CuPy/CUDA unavailable")
def test_am1_energy_cuda_smoke():
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    e = am1_energy(
        ["H", "H"],
        coords,
        coords_unit="angstrom",
        device="cuda",
        max_iter=80,
    )
    assert isinstance(e, float)
    assert np.isfinite(e)


@pytest.mark.cuda
@pytest.mark.skipif(not has_cupy() or not has_cuda_device(), reason="CuPy/CUDA unavailable")
def test_am1_energy_cuda_auto_smoke():
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    e = am1_energy(
        ["H", "H"],
        coords,
        coords_unit="angstrom",
        device="cuda",
        max_iter=80,
        fock_mode="auto",
    )
    assert isinstance(e, float)
    assert np.isfinite(e)
