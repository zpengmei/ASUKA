from __future__ import annotations

"""Centralized J/K dispatch: DF vs dense AO ERIs.

This utility allows CASSCF code to compute Coulomb/exchange matrices from
whichever 2-electron source is available (DF B-tensor or materialized AO ERIs),
avoiding scattered if/else branches throughout the driver.
"""

from typing import Any


def jk_from_2e_source(
    df_B: Any | None,
    ao_eri: Any | None,
    D: Any,
    *,
    want_J: bool = True,
    want_K: bool = True,
) -> tuple[Any | None, Any | None]:
    """Compute J and/or K from whichever 2e source is available.

    Parameters
    ----------
    df_B : Any | None
        DF B-tensor (nao, nao, naux). If not None, DF J/K is used.
    ao_eri : Any | None
        Dense AO ERI matrix (nao*nao, nao*nao). Used when df_B is None.
    D : Any
        AO density matrix (nao, nao).
    want_J : bool
        Whether to compute Coulomb matrix.
    want_K : bool
        Whether to compute exchange matrix.

    Returns
    -------
    tuple[Any | None, Any | None]
        (J, K) — each is None if the corresponding `want_*` is False.

    Raises
    ------
    ValueError
        If both df_B and ao_eri are None.
    """
    if not bool(want_J) and not bool(want_K):
        return None, None

    if df_B is not None:
        from asuka.hf import df_scf as _df_scf  # noqa: PLC0415

        return _df_scf._df_JK(df_B, D, want_J=want_J, want_K=want_K)  # noqa: SLF001

    if ao_eri is not None:
        from asuka.hf.dense_jk import dense_JK_from_eri_mat_D  # noqa: PLC0415

        return dense_JK_from_eri_mat_D(ao_eri, D, want_J=want_J, want_K=want_K)

    raise ValueError("jk_from_2e_source requires either df_B or ao_eri to be non-None")


__all__ = ["jk_from_2e_source"]
