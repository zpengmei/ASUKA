from __future__ import annotations

import numpy as np
import pytest

from asuka.semiempirical import SemiempiricalCalculator, am1_energy
from asuka.semiempirical.gpu import kernels, scf_gpu
from asuka.semiempirical.params import ANGSTROM_TO_BOHR, load_params
from asuka.semiempirical.scf import am1_scf


def _simple_h2_inputs():
    symbols = ["H", "H"]
    coords_ang = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    atomic_numbers = [1, 1]
    coords_bohr = coords_ang * ANGSTROM_TO_BOHR
    params = load_params("AM1")
    return symbols, coords_ang, atomic_numbers, coords_bohr, params


class _FakeCudaRuntimeZero:
    @staticmethod
    def getDeviceCount():
        return 0


class _FakeCudaRuntimeOne:
    @staticmethod
    def getDeviceCount():
        return 1


class _FakeCudaZero:
    runtime = _FakeCudaRuntimeZero


class _FakeCudaOne:
    runtime = _FakeCudaRuntimeOne


class _FakeCPZero:
    cuda = _FakeCudaZero


class _FakeCPOne:
    cuda = _FakeCudaOne


def test_cuda_auto_mode_prefers_w_for_small_pairs():
    assert scf_gpu._resolve_cuda_fock_mode("auto", npairs=8) == "w"


def test_cuda_auto_mode_falls_back_to_ri_for_large_pairs(monkeypatch):
    monkeypatch.setattr(scf_gpu, "_AUTO_WBLOCK_MAX_BYTES", 1024)
    assert scf_gpu._resolve_cuda_fock_mode("auto", npairs=256) == "ri"


def test_invalid_fock_mode_rejected_consistently():
    symbols, coords_ang, atomic_numbers, coords_bohr, params = _simple_h2_inputs()

    with pytest.raises(ValueError, match="fock_mode must be 'ri', 'w', or 'auto'"):
        am1_energy(symbols, coords_ang, coords_unit="angstrom", fock_mode="bad")

    calc = SemiempiricalCalculator(method="AM1")
    with pytest.raises(ValueError, match="fock_mode must be 'ri', 'w', or 'auto'"):
        calc.energy(symbols, coords_ang, coords_unit="angstrom", fock_mode="bad")

    with pytest.raises(ValueError, match="fock_mode must be 'ri', 'w', or 'auto'"):
        am1_scf(atomic_numbers, coords_bohr, params, fock_mode="bad")


def test_get_fock_kernels_requires_cupy(monkeypatch):
    monkeypatch.setattr(kernels, "_KERNEL_CACHE", None)
    monkeypatch.setattr(kernels, "_import_cupy", lambda: None)
    with pytest.raises(RuntimeError, match="CuPy is required for semiempirical CUDA kernels"):
        kernels.get_fock_kernels()


def test_get_fock_kernels_requires_cuda_device(monkeypatch):
    monkeypatch.setattr(kernels, "_KERNEL_CACHE", None)
    monkeypatch.setattr(kernels, "_import_cupy", lambda: _FakeCPZero)
    with pytest.raises(RuntimeError, match="No CUDA device is visible to CuPy"):
        kernels.get_fock_kernels()


def test_missing_kernel_source_path_reports_packaging_error(monkeypatch, tmp_path):
    missing = tmp_path / "missing_kernels.cu"
    monkeypatch.setattr(kernels, "_kernel_source_path", lambda: missing)
    with pytest.raises(RuntimeError, match="Missing AM1 CUDA kernel source file"):
        kernels.ensure_kernel_source_available()


def test_am1_scf_cuda_preflight_no_cupy(monkeypatch):
    _, _, atomic_numbers, coords_bohr, params = _simple_h2_inputs()
    monkeypatch.setattr(scf_gpu, "_import_cupy", lambda: None)
    with pytest.raises(RuntimeError, match="CuPy is required for device='cuda'"):
        scf_gpu.am1_scf_cuda(atomic_numbers=atomic_numbers, coords_bohr=coords_bohr, params=params)


def test_am1_scf_cuda_preflight_no_device(monkeypatch):
    _, _, atomic_numbers, coords_bohr, params = _simple_h2_inputs()
    monkeypatch.setattr(scf_gpu, "_import_cupy", lambda: _FakeCPZero)
    with pytest.raises(RuntimeError, match="No CUDA device is visible to CuPy"):
        scf_gpu.am1_scf_cuda(atomic_numbers=atomic_numbers, coords_bohr=coords_bohr, params=params)


def test_am1_scf_cuda_preflight_missing_kernel_source(monkeypatch):
    _, _, atomic_numbers, coords_bohr, params = _simple_h2_inputs()

    def _raise_missing_kernel():
        raise RuntimeError("Missing AM1 CUDA kernel source file")

    monkeypatch.setattr(scf_gpu, "_import_cupy", lambda: _FakeCPOne)
    monkeypatch.setattr(scf_gpu, "ensure_kernel_source_available", _raise_missing_kernel)
    with pytest.raises(RuntimeError, match="Missing AM1 CUDA kernel source file"):
        scf_gpu.am1_scf_cuda(atomic_numbers=atomic_numbers, coords_bohr=coords_bohr, params=params)
