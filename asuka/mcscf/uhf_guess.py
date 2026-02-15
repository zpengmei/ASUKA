from __future__ import annotations

"""Utilities for using UHF orbitals as a spatial-orbital initial guess.

ASUKA's GUGA-based CASCI/CASSCF solvers operate in a *single spatial orbital*
basis. This helper converts UHF orbitals (alpha/beta MO coefficients) into a
single spatial MO set via natural orbitals of the total density:

    D = D_alpha + D_beta

The resulting spatial orbitals are S-orthonormal and ordered by descending
natural occupation.

"""

from typing import Any

import numpy as np

from asuka.hf import df_scf as _df_scf


def spatialize_uhf_mo_coeff(
    *,
    S_ao: Any,
    mo_coeff: tuple[Any, Any],
    mo_occ: tuple[Any, Any],
) -> tuple[Any, Any]:
    """Return spatial natural orbitals from UHF (Ca,Cb) and (occ_a,occ_b).

    Parameters
    ----------
    S_ao : Any
        AO overlap matrix (nao,nao).
    mo_coeff : tuple[Any, Any]
        Tuple (Ca, Cb) with shape (nao,nmo) each.
    mo_occ : tuple[Any, Any]
        Tuple (occ_a, occ_b) with shape (nmo,) each.

    Returns
    -------
    C_spatial : Any
        Spatial MO coefficients (nao,nmo).
    occ_no : Any
        Natural orbital occupations (descending).
    """

    Ca, Cb = mo_coeff
    occ_a, occ_b = mo_occ

    xp, _is_gpu = _df_scf._get_xp(Ca, Cb, occ_a, occ_b)  # noqa: SLF001
    Ca = xp.asarray(Ca, dtype=xp.float64)
    Cb = xp.asarray(Cb, dtype=xp.float64)
    occ_a = xp.asarray(occ_a, dtype=xp.float64).ravel()
    occ_b = xp.asarray(occ_b, dtype=xp.float64).ravel()
    S = xp.asarray(S_ao, dtype=xp.float64)

    if Ca.ndim != 2 or Cb.ndim != 2:
        raise ValueError("mo_coeff must be (Ca,Cb) with 2D arrays")
    if Ca.shape != Cb.shape:
        raise ValueError("Ca/Cb shape mismatch")
    nao, nmo = map(int, Ca.shape)
    if S.shape != (nao, nao):
        raise ValueError("S_ao shape mismatch with mo_coeff")
    if occ_a.shape != (nmo,) or occ_b.shape != (nmo,):
        raise ValueError("mo_occ shape mismatch with mo_coeff")

    Da = (Ca * occ_a[None, :]) @ Ca.T
    Db = (Cb * occ_b[None, :]) @ Cb.T
    D = Da + Db
    D = 0.5 * (D + D.T)

    X = _df_scf._orthogonalizer_from_S(S)  # noqa: SLF001
    D_orth = X.T @ D @ X
    D_orth = 0.5 * (D_orth + D_orth.T)

    occ_no, U = xp.linalg.eigh(D_orth)
    idx = xp.argsort(occ_no)[::-1]
    occ_no = occ_no[idx]
    U = U[:, idx]

    C_spatial = X @ U

    # Preserve float64 dtype explicitly (esp. for some CuPy eigh paths).
    C_spatial = xp.asarray(C_spatial, dtype=xp.float64)
    occ_no = xp.asarray(occ_no, dtype=xp.float64)
    return C_spatial, occ_no


__all__ = ["spatialize_uhf_mo_coeff"]

