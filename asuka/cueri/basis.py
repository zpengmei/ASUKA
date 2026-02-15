from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BasisSoA:
    """Structure of Arrays (SoA) representation for a basis set of s-type shells.

    This class provides a minimal, optimized data layout for handling basis sets composed
    exclusively of s-shells (l=0). It stores primitive data and shell metadata in contiguous
    arrays to facilitate efficient processing.

    Parameters
    ----------
    shell_cxyz : np.ndarray
        Cartesian coordinates of shell centers. Shape: `(nShell, 3)`.
    shell_prim_start : np.ndarray
        Starting index of primitives for each shell. Shape: `(nShell,)`.
    shell_nprim : np.ndarray
        Number of primitives each shell is contraction of. Shape: `(nShell,)`.
    shell_l : np.ndarray
        Angular momentum of each shell. Must be all zeros (s-shells). Shape: `(nShell,)`.
    prim_exp : np.ndarray
        Exponents of the unnormalized primitive Gaussian functions `exp(-alpha * r^2)`.
        Shape: `(nPrim,)`.
    prim_coef : np.ndarray
        Contraction coefficients, including primitive normalization factors.
        Shape: `(nPrim,)`.
    source_bas_id : np.ndarray | None, optional
        Mapping to source basis ID, if applicable. Shape: `(nShell,)`.
    source_ctr_id : np.ndarray | None, optional
        Mapping to source contraction ID, if applicable. Shape: `(nShell,)`.

    Notes
    -----
    - This class enforces that all `shell_l` entries are 0.
    """

    def __post_init__(self) -> None:
        if self.shell_cxyz.dtype != np.float64:
            raise TypeError("shell_cxyz must be float64")
        if self.shell_cxyz.ndim != 2 or self.shell_cxyz.shape[1] != 3:
            raise ValueError("shell_cxyz must have shape (nShell, 3)")
        n_shell = int(self.shell_cxyz.shape[0])
        for name, arr, dt in (
            ("shell_prim_start", self.shell_prim_start, np.int32),
            ("shell_nprim", self.shell_nprim, np.int32),
            ("shell_l", self.shell_l, np.int32),
        ):
            if arr.dtype != dt:
                raise TypeError(f"{name} must be {dt}")
            if arr.shape != (n_shell,):
                raise ValueError(f"{name} must have shape (nShell,)")
        if self.prim_exp.dtype != np.float64 or self.prim_coef.dtype != np.float64:
            raise TypeError("prim_exp/prim_coef must be float64")
        if self.prim_exp.shape != self.prim_coef.shape or self.prim_exp.ndim != 1:
            raise ValueError("prim_exp and prim_coef must be 1D arrays with identical shape")
        if np.any(self.shell_l != 0):
            raise ValueError("BasisSoA currently supports s shells only (shell_l must be all zeros)")


__all__ = ["BasisSoA"]
