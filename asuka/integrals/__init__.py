"""Integral backends and helpers.

This subpackage hosts cuGUGA-owned interfaces for integral generation.

Policy
------
- cuERI is the production integral backend.
- External interoperability helpers (where present) are intended for parity tests /
  benchmarks and should not be relied on for production workflows.
"""

from __future__ import annotations

from .cueri_df import CuERIDFConfig, build_df_B_from_cueri_packed_bases
from .int1e_cart import (
    Int1eDerivResult,
    Int1eResult,
    build_dS_cart,
    build_dT_cart,
    build_dV_cart,
    build_int1e_cart,
    build_int1e_cart_deriv,
    build_S_cart,
    build_T_cart,
    build_V_cart,
    nao_cart_from_basis,
    shell_to_atom_map,
)

__all__ = [
    "CuERIDFConfig",
    "Int1eDerivResult",
    "Int1eResult",
    "build_df_B_from_cueri_packed_bases",
    "build_S_cart",
    "build_T_cart",
    "build_V_cart",
    "build_dS_cart",
    "build_dT_cart",
    "build_dV_cart",
    "build_int1e_cart",
    "build_int1e_cart_deriv",
    "nao_cart_from_basis",
    "shell_to_atom_map",
]
