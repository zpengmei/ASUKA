from __future__ import annotations

"""Exchange-correlation functional package for KS-DFT.

Provides meta-GGA functionals (MN15, M06 family) with GPU-accelerated
numerical integration via CuPy.
"""

from .eval_xc import eval_xc, eval_xc_sp, eval_xc_u
from .functional import FunctionalSpec, get_functional
from .nuc_grad import XCNucGradResult, build_vxc_nuc_grad, build_vxc_nuc_grad_from_mol
from .nuc_grad_fd import XCNucGradFDResult, build_vxc_nuc_grad_fd, build_vxc_nuc_grad_fd_from_mol
from .numint import build_vxc, build_vxc_u

__all__ = [
    "FunctionalSpec",
    "XCNucGradFDResult",
    "XCNucGradResult",
    "build_vxc",
    "build_vxc_nuc_grad",
    "build_vxc_nuc_grad_fd",
    "build_vxc_nuc_grad_fd_from_mol",
    "build_vxc_nuc_grad_from_mol",
    "build_vxc_u",
    "eval_xc",
    "eval_xc_sp",
    "eval_xc_u",
    "get_functional",
]
