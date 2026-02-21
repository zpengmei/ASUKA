from __future__ import annotations

import os
import shutil
import subprocess
import sys

from setuptools.command.build_ext import build_ext as _build_ext
from setuptools import Extension, setup


def _bool_env(key: str) -> bool:
    val = os.environ.get(key, "").strip().lower()
    return val not in ("", "0", "false", "no", "off")


def _int_env(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    if raw == "":
        return int(default)
    try:
        out = int(raw)
    except ValueError as e:
        raise SystemExit(f"{key} must be an integer, got: {raw!r}") from e
    if out < 0:
        raise SystemExit(f"{key} must be >= 0, got: {out}")
    return out


def _find_nvcc() -> str | None:
    env_nvcc = os.environ.get("GUGA_CUDA_NVCC") or os.environ.get("CUERI_CUDA_NVCC")
    env_nvcc = env_nvcc or os.environ.get("CUDACXX") or os.environ.get("NVCC")
    if env_nvcc:
        expanded = os.path.abspath(os.path.expanduser(env_nvcc))
        if os.path.exists(expanded):
            return expanded

    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc

    for path in (
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13.1/bin/nvcc",
        "/usr/local/cuda-13.0/bin/nvcc",
        "/usr/local/cuda-12.8/bin/nvcc",
        "/usr/local/cuda-12.6/bin/nvcc",
    ):
        if os.path.exists(path):
            return path
    return None


def _epq_ext() -> Extension:
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise SystemExit("NumPy is required to build asuka._epq_cy") from e

    repo_root = os.path.abspath(os.path.dirname(__file__))
    extra_compile_args = ["-O3", "-std=c++11"]
    extra_link_args: list[str] = []
    define_macros: list[tuple[str, str]] = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

    if _bool_env("GUGA_USE_OPENMP"):
        if sys.platform == "darwin":  # pragma: no cover
            raise SystemExit("GUGA_USE_OPENMP=1 is not supported by this build on macOS.")
        extra_compile_args.append("-fopenmp")
        extra_link_args.append("-fopenmp")
        define_macros.append(("GUGA_USE_OPENMP", "1"))

    return Extension(
        "asuka._epq_cy",
        sources=["asuka/_epq_cy.pyx"],
        include_dirs=[np.get_include(), repo_root],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    )


def _int1e_ext() -> Extension:
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise SystemExit("NumPy is required to build asuka.integrals._int1e_cart_cy") from e

    extra_compile_args = ["-O3"]
    define_macros: list[tuple[str, str]] = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

    return Extension(
        "asuka.integrals._int1e_cart_cy",
        sources=["asuka/integrals/_int1e_cart_cy.pyx"],
        include_dirs=[np.get_include()],
        language="c",
        extra_compile_args=extra_compile_args,
        define_macros=define_macros,
    )


def _cueri_eri_cpu_ext() -> Extension:
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise SystemExit("NumPy is required to build asuka.cueri._eri_rys_cpu") from e

    extra_compile_args = ["-O3"]
    extra_link_args: list[str] = []
    define_macros: list[tuple[str, str]] = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

    if _bool_env("CUERI_USE_OPENMP"):
        if sys.platform == "darwin":  # pragma: no cover
            raise SystemExit("CUERI_USE_OPENMP=1 is not supported by this build on macOS.")
        extra_compile_args.append("-fopenmp")
        extra_link_args.append("-fopenmp")
        define_macros.append(("CUERI_USE_OPENMP", "1"))

    return Extension(
        "asuka.cueri._eri_rys_cpu",
        sources=["asuka/cueri/_eri_rys_cpu.pyx"],
        include_dirs=[np.get_include()],
        language="c",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    )


def _cueri_pair_coeff_cpu_ext() -> Extension:
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise SystemExit("NumPy is required to build asuka.cueri._pair_coeff_cpu") from e

    extra_compile_args = ["-O3"]
    extra_link_args: list[str] = []
    define_macros: list[tuple[str, str]] = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

    if _bool_env("CUERI_USE_OPENMP"):
        if sys.platform == "darwin":  # pragma: no cover
            raise SystemExit("CUERI_USE_OPENMP=1 is not supported by this build on macOS.")
        extra_compile_args.append("-fopenmp")
        extra_link_args.append("-fopenmp")
        define_macros.append(("CUERI_USE_OPENMP", "1"))

    return Extension(
        "asuka.cueri._pair_coeff_cpu",
        sources=["asuka/cueri/_pair_coeff_cpu.pyx"],
        include_dirs=[np.get_include()],
        language="c",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    )


def _cythonize_exts(exts: list[Extension]) -> list[Extension]:
    try:
        from Cython.Build import cythonize
    except Exception as e:  # pragma: no cover
        raise SystemExit("Cython is required to build asuka Cython extensions.") from e

    cueri_cpu_max_l = _int_env("CUERI_CPU_MAX_L", 6)
    return cythonize(
        exts,
        compiler_directives={"language_level": "3"},
        compile_time_env={"CY_K_LMAX": int(cueri_cpu_max_l)},
        annotate=False,
        build_dir=os.path.join("build", "cython"),
    )


class build_ext(_build_ext):
    def run(self) -> None:
        super().run()

        if _bool_env("ASUKA_SKIP_CUDA_EXT"):
            return

        nvcc = _find_nvcc()
        if nvcc is None:
            if _bool_env("ASUKA_REQUIRE_CUDA_EXT"):
                raise SystemExit(
                    "ASUKA_REQUIRE_CUDA_EXT=1 was set, but nvcc was not found. "
                    "Install a CUDA toolkit or set ASUKA_SKIP_CUDA_EXT=1 for CPU-only installs."
                )
            print(
                "ASUKA: nvcc not found; skipping CUDA extension build. "
                "Set ASUKA_REQUIRE_CUDA_EXT=1 to make this a hard error.",
                file=sys.stderr,
            )
            return

        repo_root = os.path.abspath(os.path.dirname(__file__))
        build_temp = os.path.abspath(getattr(self, "build_temp", os.path.join(repo_root, "build", "temp")))
        build_lib = os.path.abspath(getattr(self, "build_lib", os.path.join(repo_root, "build", "lib")))

        env = os.environ.copy()

        if bool(getattr(self, "inplace", False)):
            guga_out = os.path.join(repo_root, "asuka")
            cueri_out = os.path.join(repo_root, "asuka", "cueri")
        else:
            guga_out = os.path.join(build_lib, "asuka")
            cueri_out = os.path.join(build_lib, "asuka", "cueri")

        os.makedirs(guga_out, exist_ok=True)
        os.makedirs(cueri_out, exist_ok=True)

        env["GUGA_CUDA_EXT_OUTPUT_DIR"] = guga_out

        subprocess.check_call([sys.executable, "-m", "asuka.build.guga_cuda_ext"], cwd=repo_root, env=env)
        subprocess.check_call([sys.executable, "-m", "asuka.build.guga_cuda_linalg_ext"], cwd=repo_root, env=env)
        subprocess.check_call([sys.executable, "-m", "asuka.build.caspt2_cuda_ext"], cwd=repo_root, env=env)

        cueri_build_dir = os.path.join(build_temp, "cueri_cuda_ext")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "asuka.cueri.build_cuda_ext",
                "--build-dir",
                cueri_build_dir,
                "--out-dir",
                cueri_out,
            ],
            cwd=repo_root,
            env=env,
        )


setup(
    ext_modules=_cythonize_exts([_epq_ext(), _int1e_ext(), _cueri_eri_cpu_ext(), _cueri_pair_coeff_cpu_ext()]),
    cmdclass={"build_ext": build_ext},
)
