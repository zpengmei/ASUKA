r"""Shared XMS-CASPT2 utilities (no heavy dependencies).

This module contains lightweight helper functions for XMS-CASPT2 that
have no dependency on heavy compute backends (CUDA, RDM builders, etc.).

Mathematical Background
-----------------------
In XMS-CASPT2, the reference states are rotated by a unitary matrix
:math:`U_0` obtained by diagonalising the model-space Fock operator.
The effective Hamiltonian must be corrected for this rotation:

.. math::

    H_{\text{eff}}^{\text{XMS}} = H_{\text{eff}} - \text{diag}(E_{\text{ref}})
        + U_0^T \, \text{diag}(E_{\text{ref}}) \, U_0

This ensures that the diagonal of :math:`H_{\text{eff}}^{\text{XMS}}`
reflects the reference energies in the rotated basis rather than the
original SA-CASSCF basis.
"""
from __future__ import annotations

import numpy as np


def _apply_xms_reference_rotation(
    *, heff: np.ndarray, e_ref_list: list[float], u0: np.ndarray
) -> np.ndarray:
    """Apply XMS reference-rotation correction to the effective Hamiltonian.

    H_eff_XMS = H_eff - diag(E_ref) + U0^T @ diag(E_ref) @ U0
    """
    h = np.asarray(heff, dtype=np.float64)
    u = np.asarray(u0, dtype=np.float64)
    e_ref = np.asarray(e_ref_list, dtype=np.float64).ravel()
    nstates = int(e_ref.size)
    if h.shape != (nstates, nstates):
        raise ValueError("heff shape mismatch with e_ref_list")
    if u.shape != (nstates, nstates):
        raise ValueError("u0 shape mismatch with e_ref_list")
    d_ref = np.diag(e_ref)
    h_ref_rot = np.asarray(u.T @ d_ref @ u, dtype=np.float64)
    return np.asarray(h - d_ref + h_ref_rot, dtype=np.float64, order="C")
