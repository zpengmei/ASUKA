from __future__ import annotations

import json
from pathlib import Path

import pytest

from asuka.semiempirical.gpu.kernels import (
    ensure_gradient_kernel_sources_available,
    ensure_kernel_source_available,
)
from asuka.semiempirical.params import load_params


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_packaging_metadata_includes_semiempirical_runtime_assets():
    root = _repo_root()
    if not (root / "pyproject.toml").is_file() or not (root / "MANIFEST.in").is_file():
        pytest.skip("Source-tree packaging metadata is unavailable in this environment")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")

    assert "semiempirical/data/*.json" in pyproject
    assert "semiempirical/gpu/cuda/*.cu" in pyproject
    assert "semiempirical/gpu/cuda/*.cuh" in pyproject
    assert "recursive-include asuka/semiempirical/data *.json" in manifest
    assert "recursive-include asuka/semiempirical/gpu/cuda *.cu *.cuh" in manifest


def test_runtime_assets_exist_and_are_readable():
    root = _repo_root()
    kernel_src = root / "asuka" / "semiempirical" / "gpu" / "cuda" / "fock_kernels.cu"
    grad_src = root / "asuka" / "semiempirical" / "gpu" / "cuda" / "gradient_kernels.cu"
    grad_hdr = root / "asuka" / "semiempirical" / "gpu" / "cuda" / "pair_math.cuh"
    am1_json = root / "asuka" / "semiempirical" / "data" / "am1_params.json"
    pm7_json = root / "asuka" / "semiempirical" / "data" / "pm7_params.json"

    assert kernel_src.is_file()
    assert grad_src.is_file()
    assert grad_hdr.is_file()
    assert am1_json.is_file()
    assert pm7_json.is_file()

    code = kernel_src.read_text(encoding="utf-8")
    assert "onecenter_fock_kernel" in code
    assert "twocenter_fock_ri_44_kernel" in code
    grad_code = grad_src.read_text(encoding="utf-8")
    assert "am1_grad_pair_44_kernel" in grad_code
    hdr_code = grad_hdr.read_text(encoding="utf-8")
    assert "struct Dual3" in hdr_code

    am1_data = json.loads(am1_json.read_text(encoding="utf-8"))
    pm7_data = json.loads(pm7_json.read_text(encoding="utf-8"))
    assert am1_data["name"] == "AM1"
    assert pm7_data["name"] == "PM7"


def test_runtime_resource_loading_uses_packaged_paths():
    kernel_src = ensure_kernel_source_available()
    assert kernel_src.name == "fock_kernels.cu"
    assert kernel_src.is_file()
    grad_src, grad_hdr = ensure_gradient_kernel_sources_available()
    assert grad_src.name == "gradient_kernels.cu"
    assert grad_hdr.name == "pair_math.cuh"
    assert grad_src.is_file()
    assert grad_hdr.is_file()

    am1 = load_params("AM1")
    pm7 = load_params("PM7")
    assert am1.name == "AM1"
    assert 1 in am1.elements
    assert pm7.name == "PM7"
    assert pm7.is_placeholder
