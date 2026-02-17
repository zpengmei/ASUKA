from __future__ import annotations

import numpy as np
import pytest

from asuka.nddo_core import build_pair_list
from asuka.semiempirical import SemiempiricalCalculator
from asuka.semiempirical.basis import symbol_to_Z
from asuka.semiempirical.gpu import has_cupy, has_cuda_device
from asuka.semiempirical.gpu.kernels import build_pair_buckets
from asuka.semiempirical.params import ANGSTROM_TO_BOHR

CUDA_OK = has_cupy() and has_cuda_device()

_CASES = [
    (
        "h2_11",
        ["H", "H"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float),
        "11",
    ),
    (
        "hcn_14",
        ["H", "C", "N"],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.06],
                [0.0, 0.0, 2.22],
            ],
            dtype=float,
        ),
        "14",
    ),
    (
        "ch4_41",
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
        "41",
    ),
    (
        "co_44",
        ["C", "O"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]], dtype=float),
        "44",
    ),
]


def _run_case(symbols, coords_ang, device: str, fock_mode: str):
    calc = SemiempiricalCalculator(method="AM1", charge=0, device=device)
    return calc.energy(
        symbols,
        coords_ang,
        coords_unit="angstrom",
        max_iter=120,
        conv_tol=1e-9,
        fock_mode=fock_mode,
    )


def _assert_structural_invariants(result) -> None:
    assert result.converged
    assert np.isfinite(result.energy_total)
    assert np.isfinite(result.energy_electronic)
    assert np.isfinite(result.energy_core)
    assert np.all(np.isfinite(result.F))
    assert np.all(np.isfinite(result.P))
    assert np.allclose(result.F, result.F.T, atol=1e-10)
    assert np.allclose(result.P, result.P.T, atol=1e-10)


def test_bucket_coverage_cases_cover_11_14_41_44():
    seen = set()
    for _, symbols, coords_ang, required_bucket in _CASES:
        atomic_numbers = [symbol_to_Z(sym) for sym in symbols]
        coords_bohr = np.asarray(coords_ang, dtype=float) * ANGSTROM_TO_BOHR
        pair_i, pair_j, _, _ = build_pair_list(coords_bohr)
        buckets = build_pair_buckets(atomic_numbers, pair_i, pair_j)
        assert len(buckets[required_bucket]) > 0
        for key in ("11", "14", "41", "44"):
            if len(buckets[key]) > 0:
                seen.add(key)
    assert seen == {"11", "14", "41", "44"}


@pytest.mark.cuda
@pytest.mark.skipif(not CUDA_OK, reason="CuPy/CUDA unavailable")
@pytest.mark.parametrize("case_name,symbols,coords_ang,_bucket", _CASES)
@pytest.mark.parametrize("fock_mode", ["ri", "w"])
def test_am1_cpu_cuda_parity(case_name, symbols, coords_ang, _bucket, fock_mode):
    cpu = _run_case(symbols, coords_ang, device="cpu", fock_mode=fock_mode)
    gpu = _run_case(symbols, coords_ang, device="cuda", fock_mode=fock_mode)

    _assert_structural_invariants(cpu)
    _assert_structural_invariants(gpu)
    assert abs(cpu.energy_total - gpu.energy_total) <= 1e-6
    assert abs(cpu.energy_electronic - gpu.energy_electronic) <= 1e-6
    assert abs(cpu.energy_core - gpu.energy_core) <= 1e-6
    assert abs(cpu.n_iter - gpu.n_iter) <= 3

    dF = gpu.F - cpu.F
    dP = gpu.P - cpu.P
    assert float(np.max(np.abs(dF))) <= 1e-5
    assert float(np.max(np.abs(dP))) <= 1e-5
    assert float(np.linalg.norm(dF)) <= 1e-4
    assert float(np.linalg.norm(dP)) <= 1e-4


@pytest.mark.cuda
@pytest.mark.skipif(not CUDA_OK, reason="CuPy/CUDA unavailable")
@pytest.mark.parametrize("case_name,symbols,coords_ang,_bucket", _CASES)
def test_am1_cuda_ri_w_internal_parity(case_name, symbols, coords_ang, _bucket):
    out_ri = _run_case(symbols, coords_ang, device="cuda", fock_mode="ri")
    out_w = _run_case(symbols, coords_ang, device="cuda", fock_mode="w")

    _assert_structural_invariants(out_ri)
    _assert_structural_invariants(out_w)
    assert abs(out_ri.energy_total - out_w.energy_total) <= 1e-6
    assert abs(out_ri.energy_electronic - out_w.energy_electronic) <= 1e-6
    assert abs(out_ri.energy_core - out_w.energy_core) <= 1e-6

    dF = out_ri.F - out_w.F
    dP = out_ri.P - out_w.P
    assert float(np.max(np.abs(dF))) <= 1e-5
    assert float(np.max(np.abs(dP))) <= 1e-5
    assert float(np.linalg.norm(dF)) <= 1e-4
    assert float(np.linalg.norm(dP)) <= 1e-4


@pytest.mark.cuda
@pytest.mark.skipif(not CUDA_OK, reason="CuPy/CUDA unavailable")
def test_am1_cuda_run_to_run_energy_stability():
    symbols = ["H", "H"]
    coords_ang = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    energies = []
    for _ in range(3):
        out = _run_case(symbols, coords_ang, device="cuda", fock_mode="ri")
        _assert_structural_invariants(out)
        energies.append(float(out.energy_total))
    span = max(energies) - min(energies)
    assert span <= 1e-6
