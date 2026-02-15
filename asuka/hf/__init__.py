"""Hartree–Fock (SCF) helpers.

This subpackage operates on already-built integrals (typically in the AO basis).
It can be used as a building block for workflows where integrals are supplied by
cuERI or other integral backends.
"""

from __future__ import annotations

from .dense_eri import DenseAOERIResult, build_ao_eri_dense, estimate_dense_eri_nbytes
from .dense_scf import rhf_dense, rohf_dense, uhf_dense
from .df_scf import SCFResult, rhf_df, rohf_df, uhf_df
from asuka.integrals.cueri_df import CuERIDFConfig, build_df_B_from_cueri_packed_bases

__all__ = [
    "SCFResult",
    "CuERIDFConfig",
    "DenseAOERIResult",
    "build_df_B_from_cueri_packed_bases",
    "build_ao_eri_dense",
    "estimate_dense_eri_nbytes",
    "rhf_dense",
    "rohf_dense",
    "uhf_dense",
    "rhf_df",
    "rohf_df",
    "uhf_df",
]
