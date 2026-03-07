from __future__ import annotations

"""Exchange-correlation functional package for KS-DFT.

Provides meta-GGA functionals (MN15, M06 family) with GPU-accelerated
numerical integration via CuPy.
"""

from .eval_xc import eval_xc
from .functional import FunctionalSpec, get_functional
from .numint import build_vxc

__all__ = [
    "FunctionalSpec",
    "build_vxc",
    "eval_xc",
    "get_functional",
]
