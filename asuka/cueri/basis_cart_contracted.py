from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BasisCartContractedSoA:
    """Structure of Arrays (SoA) representation for a Cartesian basis set with general contractions.

    This class provides a storage layout that explicitly supports generally contracted shells,
    where a single shell (defined by L, center, and primitives) can contain multiple contraction
    sets (columns of coefficients).

    Parameters
    ----------
    shell_cxyz : np.ndarray
        Cartesian coordinates of shell centers. Shape: `(nShell, 3)`.
    shell_prim_start : np.ndarray
        Start index of primitives for this shell in `prim_exp`. Shape: `(nShell,)`.
    shell_nprim : np.ndarray
        Number of primitives for this shell. Shape: `(nShell,)`.
    shell_l : np.ndarray
        Angular momentum of each shell. Shape: `(nShell,)`.
    shell_ao_start : np.ndarray
        Starting AO index for the *first* contraction of this shell.
        Subsequent contractions are assumed to follow contiguously. Shape: `(nShell,)`.
    shell_nctr : np.ndarray
        Number of contraction sets (columns) for this shell. Shape: `(nShell,)`.
    shell_coef_start : np.ndarray
        Starting index in `prim_coef_flat` for this shell's coefficients. Shape: `(nShell,)`.
    prim_exp : np.ndarray
        Exponents of the unnormalized primitive Gaussians. Shape: `(nPrim,)`.
    prim_coef_flat : np.ndarray
        Flattened contraction coefficients (including primitive normalization).
        Stored in primitive-major order within a shell: `(prim, ctr)`.
        Shape: `(sum(nprim * nctr),)`.
    source_bas_id : np.ndarray | None, optional
        Source basis ID mapping. Shape: `(nShell,)`.

    Notes
    -----
    - This is the "contracted" counterpart to :class:`~cueri.basis_cart.BasisCartSoA`.
    - Flattened coefficients allow efficient storage of ragged contraction structures.
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
            ("shell_ao_start", self.shell_ao_start, np.int32),
            ("shell_nctr", self.shell_nctr, np.int32),
            ("shell_coef_start", self.shell_coef_start, np.int32),
        ):
            if arr.dtype != dt:
                raise TypeError(f"{name} must be {dt}")
            if arr.shape != (n_shell,):
                raise ValueError(f"{name} must have shape (nShell,)")

        if self.prim_exp.dtype != np.float64:
            raise TypeError("prim_exp must be float64")
        if self.prim_exp.ndim != 1:
            raise ValueError("prim_exp must be a 1D array")

        if self.prim_coef_flat.dtype != np.float64:
            raise TypeError("prim_coef_flat must be float64")
        if self.prim_coef_flat.ndim != 1:
            raise ValueError("prim_coef_flat must be a 1D array")

        if np.any(self.shell_nprim < 0):
            raise ValueError("shell_nprim must be >= 0")
        if np.any(self.shell_nctr < 1):
            raise ValueError("shell_nctr must be >= 1")
        if np.any(self.shell_l < 0):
            raise ValueError("shell_l must be >= 0")

        # Validate coefficient storage size matches offsets + nprim*nctr.
        expect_coef = int(np.sum(self.shell_nprim.astype(np.int64) * self.shell_nctr.astype(np.int64)))
        if int(self.prim_coef_flat.shape[0]) != expect_coef:
            raise ValueError(f"prim_coef_flat has length {self.prim_coef_flat.shape[0]}, expected {expect_coef}")

        if self.source_bas_id is not None:
            if np.asarray(self.source_bas_id).dtype != np.int32:
                raise TypeError("source_bas_id must be int32 when provided")
            if np.asarray(self.source_bas_id).shape != (n_shell,):
                raise ValueError("source_bas_id must have shape (nShell,)")


__all__ = ["BasisCartContractedSoA"]

