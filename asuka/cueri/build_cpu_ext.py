#!/usr/bin/env python
from __future__ import annotations

import os
import sys


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


def main() -> None:
    cueri_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(cueri_dir, "..", ".."))
    os.chdir(repo_root)

    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise SystemExit("NumPy is required to build the cuERI CPU extensions") from e

    from setuptools import Extension, setup

    use_openmp_env = os.environ.get("CUERI_USE_OPENMP", "").strip().lower()
    use_openmp = use_openmp_env not in ("", "0", "false", "no", "off")

    try:
        from Cython.Build import cythonize
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Cython is required to build asuka.cueri._eri_rys_cpu. "
            "Install with: python -m pip install Cython"
        ) from e

    extra_compile_args = ["-O3"]
    extra_link_args: list[str] = []
    define_macros: list[tuple[str, str]] = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    if use_openmp:
        if sys.platform == "darwin":  # pragma: no cover
            raise SystemExit("CUERI_USE_OPENMP=1 is not supported by this build helper on macOS.")
        extra_compile_args.append("-fopenmp")
        extra_link_args.append("-fopenmp")
        define_macros.append(("CUERI_USE_OPENMP", "1"))

    ext_eri = Extension(
        "asuka.cueri._eri_rys_cpu",
        sources=[os.path.join("asuka", "cueri", "_eri_rys_cpu.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    )

    ext_pair = Extension(
        "asuka.cueri._pair_coeff_cpu",
        sources=[os.path.join("asuka", "cueri", "_pair_coeff_cpu.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    )

    setup(
        name="asuka_cueri_cpu_ext",
        ext_modules=cythonize(
            [ext_eri, ext_pair],
            compiler_directives={"language_level": "3"},
            compile_time_env={"CY_K_LMAX": _int_env("CUERI_CPU_MAX_L", 6)},
            annotate=False,
            build_dir=os.path.join("build", "cython"),
        ),
        script_args=sys.argv[1:],
    )


if __name__ == "__main__":
    main()
