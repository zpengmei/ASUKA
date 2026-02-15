#!/usr/bin/env python
from __future__ import annotations

import os
import sys


def main() -> None:
    build_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(build_dir, "..", ".."))
    os.chdir(repo_root)

    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise SystemExit("NumPy is required to build the int1e Cython extension") from e

    from setuptools import Extension, setup

    try:
        from Cython.Build import cythonize
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Cython is required to build asuka.integrals._int1e_cart_cy. "
            "Install with: python -m pip install Cython"
        ) from e

    ext = Extension(
        "asuka.integrals._int1e_cart_cy",
        sources=[os.path.join("asuka", "integrals", "_int1e_cart_cy.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )

    setup(
        name="asuka_int1e_ext",
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

