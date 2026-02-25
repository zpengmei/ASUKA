"""Shared XMS-CASPT2 utilities (no heavy dependencies)."""
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
