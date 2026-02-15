#!/usr/bin/env python
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _cmake_dir_for_pybind11() -> str:
    out = subprocess.check_output([sys.executable, "-m", "pybind11", "--cmakedir"], text=True)
    return out.strip()


def _cuda_root_from_nvcc(nvcc: str) -> str | None:
    nvcc = os.path.abspath(nvcc)
    root = os.path.dirname(os.path.dirname(nvcc))
    if not os.path.exists(os.path.join(root, "include")):
        return None
    if os.path.exists(os.path.join(root, "lib64")) or os.path.exists(os.path.join(root, "lib")):
        return root
    return None


def _cuda_root_override() -> str | None:
    for key in ("GUGA_CUDA_ROOT", "CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH"):
        val = os.getenv(key)
        if not val:
            continue
        root = os.path.abspath(val)
        if os.path.exists(os.path.join(root, "include")) and (
            os.path.exists(os.path.join(root, "lib64")) or os.path.exists(os.path.join(root, "lib"))
        ):
            return root
    return None


def _find_nvcc() -> str | None:
    nvcc_env = os.getenv("GUGA_CUDA_NVCC") or os.getenv("CUDACXX") or os.getenv("NVCC")
    if nvcc_env and os.path.exists(nvcc_env):
        return nvcc_env
    preferred = [
        "/usr/local/cuda-13.1/bin/nvcc",
        "/usr/local/cuda-13.0/bin/nvcc",
        "/usr/local/cuda-13/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
    ]
    for path in preferred:
        if os.path.exists(path):
            return path
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    candidates = [
        "/usr/local/cuda-12.8/bin/nvcc",
        "/usr/local/cuda-12.6/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _detect_cuda_arch() -> str | None:
    """Best-effort GPU CC detection for CMAKE_CUDA_ARCHITECTURES.

    Returns a string like "89" (sm_89) or None if detection fails.
    """
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True
            ).strip()
            line0 = out.splitlines()[0].strip() if out else ""
            m = re.match(r"^([0-9]+)[.]([0-9]+)$", line0)
            if m:
                return f"{m.group(1)}{m.group(2)}"
        except Exception:
            pass
    return None


def main() -> None:
    guga_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(guga_dir, "..", ".."))
    os.chdir(repo_root)

    try:
        import pybind11  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit("pybind11 is required to build the CUDA extension") from e

    nvcc = _find_nvcc()
    if nvcc is None:  # pragma: no cover
        raise SystemExit("nvcc was not found on PATH; a CUDA toolkit is required")
    cuda_root = _cuda_root_override() or _cuda_root_from_nvcc(nvcc)
    if cuda_root is None:  # pragma: no cover
        raise SystemExit(f"could not determine CUDA root for nvcc={nvcc!r}")

    src_dir = os.path.join(repo_root, "asuka", "cuda", "ext")
    build_dir = os.path.join(repo_root, "build", "guga_cuda_ext")
    out_dir = os.getenv("GUGA_CUDA_EXT_OUTPUT_DIR") or os.path.join(repo_root, "asuka")
    out_dir = str(Path(out_dir).expanduser().resolve())

    cache_file = os.path.join(build_dir, "CMakeCache.txt")
    if os.path.exists(cache_file):
        try:
            cache_txt = open(cache_file, "r", encoding="utf-8").read()
        except OSError:
            cache_txt = ""
        cached = None
        for line in cache_txt.splitlines():
            if line.startswith("CMAKE_CUDA_COMPILER:FILEPATH="):
                cached = line.split("=", 1)[1].strip()
                break
        if cached and cached != nvcc:
            shutil.rmtree(build_dir, ignore_errors=True)

    os.makedirs(build_dir, exist_ok=True)

    print(f"Using CUDA root: {cuda_root}", file=sys.stderr)
    print(f"Using nvcc: {nvcc}", file=sys.stderr)

    cmake_args = [
        "cmake",
        f"-S{src_dir}",
        f"-B{build_dir}",
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-Dpybind11_DIR={_cmake_dir_for_pybind11()}",
        f"-DGUGA_CUDA_EXT_OUTPUT_DIR={out_dir}",
        f"-DCMAKE_CUDA_COMPILER={nvcc}",
        f"-DCUDAToolkit_ROOT={cuda_root}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    # CMake 3.22 expects plain numeric architectures (e.g. "80" or "89"), not the newer "80-virtual" suffix.
    cuda_arch_env = os.getenv("GUGA_CUDA_ARCH")
    if cuda_arch_env is None:
        auto_arch = _detect_cuda_arch()
        cuda_arch = auto_arch or "80"
        msg = "auto-detected" if auto_arch else "default"
        print(
            f"Using CMAKE_CUDA_ARCHITECTURES={cuda_arch} ({msg}; override with GUGA_CUDA_ARCH)",
            file=sys.stderr,
        )
    else:
        cuda_arch = cuda_arch_env.strip() or "80"
    if cuda_arch:
        parts: list[str] = []
        for part in cuda_arch.split(";"):
            p = part.strip()
            for suf in ("-virtual", "-real"):
                if p.endswith(suf):
                    p = p[: -len(suf)]
            if p:
                parts.append(p)
        cuda_arch = ";".join(parts) if parts else "80"
    cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")

    extra_cfg = os.getenv("GUGA_CUDA_CMAKE_CONFIGURE_ARGS")
    if extra_cfg:
        cmake_args.extend(extra_cfg.split())

    subprocess.check_call(cmake_args)

    build_args = ["cmake", "--build", build_dir, "-j", "4"]
    extra_build = os.getenv("GUGA_CUDA_CMAKE_BUILD_ARGS")
    if extra_build:
        build_args.extend(extra_build.split())
    subprocess.check_call(build_args)


if __name__ == "__main__":
    main()
