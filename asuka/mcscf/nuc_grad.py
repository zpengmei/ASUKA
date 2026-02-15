from __future__ import annotations

"""Nuclear gradients for CASCI/CASSCF.

This module provides a default implementation based on DF
gradient kernels in :mod:`asuka.mcscf.nuc_grad_df`.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .nuc_grad_df import (
    DFNucGradResult,
    casscf_nuc_grad_df,
    casci_nuc_grad_df_relaxed,
    casci_nuc_grad_df_unrelaxed,
)


@dataclass(frozen=True)
class NucGradResult:
    """Nuclear gradient container (Eh/Bohr).

    Mirrors the historical return type from the PySCF-backed wrappers.

    Attributes
    ----------
    e_tot : float
        Total energy in Hartree.
    e_nuc : float
        Nuclear repulsion energy in Hartree.
    grad : np.ndarray
        Gradient array (natm, 3) in Eh/Bohr.
    """

    e_tot: float
    e_nuc: float
    grad: np.ndarray


def _from_df(res: DFNucGradResult) -> NucGradResult:
    return NucGradResult(
        e_tot=float(res.e_tot),
        e_nuc=float(res.e_nuc),
        grad=np.asarray(res.grad, dtype=np.float64),
    )


def casci_nuc_grad(*args: Any, **kwargs: Any) -> NucGradResult:
    """Compute CASCI nuclear gradients.

    Call pattern
    ------------
    ``casci_nuc_grad(scf_out, casci, *, relaxed=True, **df_kwargs)``

    Delegates to :func:`asuka.mcscf.nuc_grad_df.casci_nuc_grad_df_relaxed` (or
    `_unrelaxed` when ``relaxed=False``).

    Parameters
    ----------
    *args : Any
        Positional arguments (scf_out, casci).
    **kwargs : Any
        Keyword arguments passed to backend.

    Returns
    -------
    NucGradResult
        Gradient result container.

    Raises
    ------
    TypeError
        If arguments do not match the expected pattern.
    """

    if len(args) == 2:
        scf_out, casci = args
        relaxed = bool(kwargs.pop("relaxed", True))
        if relaxed:
            return _from_df(casci_nuc_grad_df_relaxed(scf_out, casci, **kwargs))
        return _from_df(casci_nuc_grad_df_unrelaxed(scf_out, casci, **kwargs))

    if len(args) == 1:
        raise TypeError(
            "casci_nuc_grad no longer supports the legacy PySCF workflow "
            "`casci_nuc_grad(mol, ...)`. Use `casci_nuc_grad(scf_out, casci, ...)` "
            "or call `casci_nuc_grad_df_relaxed/unrelaxed` directly."
        )

    raise TypeError("casci_nuc_grad expects (scf_out, casci, ...)")


def casscf_nuc_grad(*args: Any, **kwargs: Any) -> NucGradResult:
    """Compute CASSCF nuclear gradients.

    Call pattern
    ------------
    ``casscf_nuc_grad(scf_out, casscf, **df_kwargs)``

    Delegates to :func:`asuka.mcscf.nuc_grad_df.casscf_nuc_grad_df`.

    Parameters
    ----------
    *args : Any
        Positional arguments (scf_out, casscf).
    **kwargs : Any
        Keyword arguments passed to backend.

    Returns
    -------
    NucGradResult
        Gradient result container.

    Raises
    ------
    TypeError
        If arguments do not match the expected pattern.
    """

    if len(args) == 2:
        scf_out, casscf = args
        return _from_df(casscf_nuc_grad_df(scf_out, casscf, **kwargs))

    if len(args) == 1:
        raise TypeError(
            "casscf_nuc_grad no longer supports the legacy PySCF workflow "
            "`casscf_nuc_grad(mol, ...)`. Use `casscf_nuc_grad(scf_out, casscf, ...)` "
            "or call `casscf_nuc_grad_df` directly."
        )

    raise TypeError("casscf_nuc_grad expects (scf_out, casscf, ...)")


__all__ = ["NucGradResult", "casci_nuc_grad", "casscf_nuc_grad"]
