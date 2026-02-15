from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from .cart import ncart


@dataclass(frozen=True)
class BasisCartSoA:
    """Structure of Arrays (SoA) representation for a Cartesian basis set (general l).

    This class stores basis set data for shells of arbitrary angular momentum in an expanded,
    uncontracted-view format. Each logical shell in this structure corresponds to a single
    contraction component (nctr=1).

    Parameters
    ----------
    shell_cxyz : np.ndarray
        Cartesian coordinates of shell centers. Shape: `(nShell, 3)`.
    shell_prim_start : np.ndarray
        Start index in `prim_exp` and `prim_coef` for each shell. Shape: `(nShell,)`.
    shell_nprim : np.ndarray
        Number of primitives for each shell. Shape: `(nShell,)`.
    shell_l : np.ndarray
        Angular momentum of each shell. Shape: `(nShell,)`.
    shell_ao_start : np.ndarray
        Starting index of the first Cartesian AO function for each shell.
        Subsequent functions follow standard Cartesian ordering. Shape: `(nShell,)`.
    prim_exp : np.ndarray
        Exponents of the unnormalized primitive Gaussians. Shape: `(nPrim,)`.
    prim_coef : np.ndarray
        Contraction coefficients (including normalization). Shape: `(nPrim,)`.
    source_bas_id : np.ndarray | None, optional
        ID of the source basis definition, if derived from a multi-basis context.
    source_ctr_id : np.ndarray | None, optional
        ID of the contraction within the source basis.

    Notes
    -----
    - This structure is "expanded", meaning it does not support general contractions explicitly.
      Contracted shells should be pre-processed (e.g., via `expand_contracted_cart_basis`)
      to replicate shells per contraction coefficient column.
    """

    shell_cxyz: np.ndarray
    shell_prim_start: np.ndarray
    shell_nprim: np.ndarray
    shell_l: np.ndarray
    shell_ao_start: np.ndarray
    prim_exp: np.ndarray
    prim_coef: np.ndarray
    source_bas_id: np.ndarray | None = None
    source_ctr_id: np.ndarray | None = None

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
            ("shell_ao_start", self.shell_ao_start, np.int32),
        ):
            if arr.dtype != dt:
                raise TypeError(f"{name} must be {dt}")
            if arr.shape != (n_shell,):
                raise ValueError(f"{name} must have shape (nShell,)")
        if self.prim_exp.dtype != np.float64 or self.prim_coef.dtype != np.float64:
            raise TypeError("prim_exp/prim_coef must be float64")
        if self.prim_exp.shape != self.prim_coef.shape or self.prim_exp.ndim != 1:
            raise ValueError("prim_exp and prim_coef must be 1D arrays with identical shape")


__all__ = ["BasisCartSoA"]
