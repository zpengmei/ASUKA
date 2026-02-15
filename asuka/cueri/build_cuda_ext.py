#!/usr/bin/env python
from __future__ import annotations

import argparse
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
    for key in ("CUERI_CUDA_ROOT", "CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH"):
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
    nvcc_env = os.getenv("CUERI_CUDA_NVCC") or os.getenv("CUDACXX") or os.getenv("NVCC")
    if nvcc_env and os.path.exists(nvcc_env):
        return nvcc_env
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    preferred = [
        "/usr/local/cuda-13.1/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
    ]
    for path in preferred:
        if os.path.exists(path):
            return path
    return None


def _detect_cuda_arch() -> str | None:
    """Detect the compute capability of the installed NVIDIA GPU.

    Attempts to determine the CUDA compute capability (e.g., "80" for Ampere A100)
    using `nvidia-smi`. This is used to set `CMAKE_CUDA_ARCHITECTURES` if not
    explicitly provided.

    Returns
    -------
    str | None
        The compute capability string (e.g., "80", "75"), or None if detection fails.
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


def _default_build_dir(*, repo_root: Path) -> Path:
    env = os.getenv("CUERI_CUDA_BUILD_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (repo_root / "build" / "cueri_cuda_ext").resolve()


def _ensure_writable_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    try:
        path.mkdir(parents=True, exist_ok=True)
        test = path / ".cueri_write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        return path
    except Exception:
        cache = Path.home() / ".cache" / "cueri" / "build" / "cueri_cuda_ext"
        cache.mkdir(parents=True, exist_ok=True)
        return cache


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the cuERI CUDA extension (pybind11 + CMake + nvcc).")
    ap.add_argument("--build-dir", default=None, help="CMake build directory (default: CUERI_CUDA_BUILD_DIR or ./build/...)")
    ap.add_argument("--out-dir", default=None, help="Output directory for the built extension (default: package dir)")
    ap.add_argument("--nvcc", default=None, help="Path to nvcc (overrides CUERI_CUDA_NVCC/CUDACXX/NVCC)")
    ap.add_argument("--cuda-root", default=None, help="CUDA toolkit root (overrides CUERI_CUDA_ROOT/CUDAToolkit_ROOT/CUDA_HOME)")
    ap.add_argument(
        "--arch",
        default=None,
        help="CMAKE_CUDA_ARCHITECTURES, e.g. 80, 89, 90, or '80;90' (overrides CUERI_CUDA_ARCH)",
    )
    ap.add_argument("--clean", action="store_true", help="Remove the build directory before configuring")
    ap.add_argument("--fast-boys", action="store_true", help="Enable CUERI_FAST_BOYS (faster Boys moments for NROOTS<=3)")
    ap.add_argument("--jobs", type=int, default=0, help="Build parallelism for cmake --build (default: auto)")
    args = ap.parse_args(argv)

    cueri_dir = Path(__file__).resolve().parent
    repo_root = cueri_dir.parent

    if shutil.which("cmake") is None:  # pragma: no cover
        raise SystemExit("cmake is required (install with `pip install asuka[build]` or your system package manager)")

    try:
        import pybind11  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit("pybind11 is required to build the cuERI CUDA extension") from e

    nvcc = str(Path(args.nvcc).expanduser().resolve()) if args.nvcc else _find_nvcc()
    if nvcc is None:  # pragma: no cover
        raise SystemExit("nvcc was not found on PATH; a CUDA toolkit is required")
    cuda_root = (
        str(Path(args.cuda_root).expanduser().resolve())
        if args.cuda_root
        else (_cuda_root_override() or _cuda_root_from_nvcc(nvcc))
    )
    if cuda_root is None:  # pragma: no cover
        raise SystemExit(f"could not determine CUDA root for nvcc={nvcc!r}")

    src_dir = cueri_dir / "cuda" / "ext"
    if not src_dir.is_dir():  # pragma: no cover
        raise SystemExit(f"missing CUDA extension sources at {src_dir}")

    build_dir = Path(args.build_dir).expanduser().resolve() if args.build_dir else _default_build_dir(repo_root=repo_root)
    build_dir = _ensure_writable_dir(build_dir)
    if bool(args.clean) and build_dir.is_dir():
        shutil.rmtree(str(build_dir), ignore_errors=True)
        build_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else cueri_dir

    build_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using CUDA root: {cuda_root}", file=sys.stderr)
    print(f"Using nvcc: {nvcc}", file=sys.stderr)

    cmake_args = [
        "cmake",
        f"-S{src_dir}",
        f"-B{build_dir}",
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-Dpybind11_DIR={_cmake_dir_for_pybind11()}",
        f"-DCUERI_CUDA_EXT_OUTPUT_DIR={out_dir}",
        f"-DCMAKE_CUDA_COMPILER={nvcc}",
        f"-DCUDAToolkit_ROOT={cuda_root}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if bool(args.fast_boys):
        cmake_args.append("-DCUERI_FAST_BOYS=ON")

    cuda_arch_env = str(args.arch).strip() if args.arch is not None else os.getenv("CUERI_CUDA_ARCH")
    if not cuda_arch_env:
        auto_arch = _detect_cuda_arch()
        cuda_arch = auto_arch or "80"
        msg = "auto-detected" if auto_arch else "default"
        print(
            f"Using CMAKE_CUDA_ARCHITECTURES={cuda_arch} ({msg}; override with CUERI_CUDA_ARCH)",
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

    extra_cfg = os.getenv("CUERI_CUDA_CMAKE_CONFIGURE_ARGS")
    if extra_cfg:
        cmake_args.extend(extra_cfg.split())

    subprocess.check_call(cmake_args)

    jobs = int(args.jobs)
    if jobs <= 0:
        jobs = max(1, min(8, int(os.cpu_count() or 4)))

    build_args = ["cmake", "--build", str(build_dir), "-j", str(jobs)]
    extra_build = os.getenv("CUERI_CUDA_CMAKE_BUILD_ARGS")
    if extra_build:
        build_args.extend(extra_build.split())
    subprocess.check_call(build_args)


if __name__ == "__main__":
    main()
