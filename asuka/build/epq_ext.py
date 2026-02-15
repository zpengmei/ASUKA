#!/usr/bin/env python
from __future__ import annotations

import os
import sys


def main() -> None:
    guga_dir = os.path.abspath(os.path.dirname(__file__))
    # This file lives at `asuka/build/epq_ext.py`. The repo root is two levels up.
    repo_root = os.path.abspath(os.path.join(guga_dir, "..", ".."))
    os.chdir(repo_root)

    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise SystemExit("NumPy is required to build the EPQ extension") from e

    from setuptools import Extension, setup

    use_openmp_env = os.environ.get("GUGA_USE_OPENMP", "").strip().lower()
    use_openmp = use_openmp_env not in ("", "0", "false", "no", "off")

    extra_compile_args = ["-O3", "-std=c++11"]
    extra_link_args: list[str] = []
    define_macros: list[tuple[str, str]] = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    if use_openmp:
        if sys.platform == "darwin":  # pragma: no cover
            raise SystemExit(
                "GUGA_USE_OPENMP=1 is not supported by this build helper on macOS. "
                "Try building with an OpenMP-capable toolchain (e.g. llvm-openmp) and "
                "add the required flags manually."
            )
        extra_compile_args.append("-fopenmp")
        extra_link_args.append("-fopenmp")
        define_macros.append(("GUGA_USE_OPENMP", "1"))

    try:
        from Cython.Build import cythonize
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Cython is required to build asuka._epq_cy (no vendored C++ fallback). "
            "Install with: python -m pip install Cython"
        ) from e

    ext = Extension(
        "asuka._epq_cy",
        sources=[os.path.join("asuka", "_epq_cy.pyx")],
        include_dirs=[np.get_include(), repo_root],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    )

    setup(
        name="asuka_epq_ext",
        ext_modules=cythonize(
            [ext],
            compiler_directives={"language_level": "3"},
            annotate=False,
            build_dir=os.path.join("build", "cython"),
        ),
        script_args=sys.argv[1:],
    )


if __name__ == "__main__":
    main()
