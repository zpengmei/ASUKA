from __future__ import annotations

from typing import Any

import numpy as np

from asuka.integrals.cueri_df import CuERIDFConfig

from ._scf_cache import cuda_device_id_or_neg1, mol_cache_key, normalize_basis_key
from ._scf_config import df_config_key


def rhf_prep_key(
    mol,
    *,
    basis_in: Any,
    auxbasis: Any,
    expand_contractions: bool,
    df_config: CuERIDFConfig | None,
    df_layout_build: str = "mnQ",
) -> tuple[Any, ...]:
    return (
        mol_cache_key(mol),
        normalize_basis_key(basis_in),
        normalize_basis_key(auxbasis),
        bool(expand_contractions),
        df_config_key(df_config),
        str(df_layout_build).strip().lower(),
    )


def rhf_guess_key(
    mol,
    *,
    basis_in: Any,
    auxbasis: Any,
    expand_contractions: bool,
) -> tuple[Any, ...]:
    return (
        "rhf",
        mol_cache_key(mol),
        normalize_basis_key(basis_in),
        normalize_basis_key(auxbasis),
        bool(expand_contractions),
        int(cuda_device_id_or_neg1()),
    )


def copy_mo_coeff_for_cache(mo_coeff: Any):
    try:
        import cupy as cp  # noqa: PLC0415
    except Exception:
        cp = None  # type: ignore
    if cp is not None and isinstance(mo_coeff, cp.ndarray):  # type: ignore[attr-defined]
        return cp.ascontiguousarray(cp.asarray(mo_coeff, dtype=cp.float64))
    return np.asarray(mo_coeff, dtype=np.float64, order="C").copy()
