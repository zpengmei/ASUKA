from __future__ import annotations

"""Convenience orbital-rotation helpers.

This module provides:
- Pair (Givens) rotations on MO coefficients
- Cayley-parameterized rotations from packed antisymmetric parameters
- A thin convenience wrapper to restart/reoptimize CASSCF from rotated orbitals
"""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from asuka.mcscf.casscf import CASSCFResult, run_casscf
from asuka.mcscf.orbital_grad import allowed_rotation_mask, cayley_update


def _get_xp(*arrs: Any) -> Any:
    """Return numpy or cupy module depending on input arrays."""

    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None
    if cp is not None:
        for a in arrs:
            if isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
                return cp
    return np


@dataclass(frozen=True)
class RotationSpace:
    """Pack/unpack conventions for orbital-rotation parameters.

    Convention:
    - Unique variables live in a strict lower-triangle mask (i>j).
    - `unpack_to_antisym(vec)` builds an antisymmetric matrix A = L - L^T where
      L contains the packed variables in its lower triangle.

    Attributes
    ----------
    nmo : int
        Number of molecular orbitals.
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    mask : np.ndarray
        Boolean mask (nmo,nmo) for strict lower triangle.
    """

    nmo: int
    ncore: int
    ncas: int
    mask: np.ndarray  # bool (nmo,nmo), strict lower triangle

    @classmethod
    def casscf_nonredundant(cls, *, nmo: int, ncore: int, ncas: int) -> "RotationSpace":
        """Create a RotationSpace with the standard CASSCF non-redundant mask.

        Parameters
        ----------
        nmo : int
            Number of MOs.
        ncore : int
            Number of core orbitals.
        ncas : int
            Number of active orbitals.

        Returns
        -------
        RotationSpace
            Configured rotation space object.
        """
        m = allowed_rotation_mask(int(nmo), int(ncore), int(ncas))
        return cls(nmo=int(nmo), ncore=int(ncore), ncas=int(ncas), mask=np.asarray(m, dtype=bool))

    def pack_lower(self, L: Any) -> np.ndarray:
        """Extract allowed lower-triangle elements into a flat vector.

        Parameters
        ----------
        L : Any
            Input matrix.

        Returns
        -------
        np.ndarray
            Flattened vector of allowed elements.
        """
        L = np.asarray(L, dtype=np.float64)
        if L.shape != (self.nmo, self.nmo):
            raise ValueError("L shape mismatch")
        return np.asarray(L[self.mask], dtype=np.float64).ravel()

    def unpack_to_antisym(self, vec: Any) -> np.ndarray:
        """Reconstruct antisymmetric matrix from packed vector.

        Parameters
        ----------
        vec : Any
            Packed vector.

        Returns
        -------
        np.ndarray
            Antisymmetric matrix (nmo,nmo).
        """
        v = np.asarray(vec, dtype=np.float64).ravel()
        nvar = int(np.count_nonzero(self.mask))
        if int(v.size) != nvar:
            raise ValueError(f"rotation vector length mismatch: expected {nvar}, got {int(v.size)}")
        L = np.zeros((self.nmo, self.nmo), dtype=np.float64)
        L[self.mask] = v
        L = np.tril(L, k=-1)
        return L - L.T

    def mask_active_virtual_only(self) -> np.ndarray:
        """Return a strict-lower-triangle mask allowing only active-virtual rotations."""

        nmo, ncore, ncas = int(self.nmo), int(self.ncore), int(self.ncas)
        nocc = ncore + ncas
        act = np.zeros((nmo,), dtype=bool)
        virt = np.zeros((nmo,), dtype=bool)
        act[ncore:nocc] = True
        virt[nocc:] = True

        m = np.zeros((nmo, nmo), dtype=bool)
        m[np.ix_(virt, act)] = True
        m &= np.tril(np.ones((nmo, nmo), dtype=bool), k=-1)
        return m

    def mask_core_frozen(self) -> np.ndarray:
        """Return a mask dropping rotations involving core orbitals."""

        nmo, ncore = int(self.nmo), int(self.ncore)
        core = np.zeros((nmo,), dtype=bool)
        core[:ncore] = True

        m = np.asarray(self.mask, dtype=bool).copy()
        m[:, core] = False
        m[core, :] = False
        return m


def rotate_pairs(C: Any, pairs: Sequence[tuple[int, int, float]], *, degrees: bool = False) -> Any:
    """Apply sequential pair (Givens) rotations to MO coefficients.

    Each pair is (i, j, theta). For real orbitals:
      [i'] =  cosθ [i] - sinθ [j]
      [j'] =  sinθ [i] + cosθ [j]

    Parameters
    ----------
    C : Any
        MO coefficients (nao, nmo).
    pairs : Sequence[tuple[int, int, float]]
        List of (i, j, angle) tuples.
    degrees : bool, optional
        If True, angles are in degrees. Default is False (radians).

    Returns
    -------
    Any
        Rotated MO coefficients.

    Notes
    -----
    Works for NumPy or CuPy arrays.
    """

    _xp = _get_xp(C)
    C = C.copy()
    nmo = int(C.shape[1])

    for (i, j, theta) in pairs:
        i = int(i)
        j = int(j)
        if i < 0 or j < 0 or i >= nmo or j >= nmo or i == j:
            raise ValueError("invalid (i,j) in pair rotation")
        th = float(theta) * (np.pi / 180.0) if degrees else float(theta)
        c = float(np.cos(th))
        s = float(np.sin(th))

        Ci = C[:, i].copy()
        Cj = C[:, j].copy()
        C[:, i] = c * Ci - s * Cj
        C[:, j] = s * Ci + c * Cj

    return C


def rotate_kappa_vec(C: Any, space: RotationSpace, kappa_vec: Any, *, method: str = "cayley") -> Any:
    """Rotate orbitals by a packed vector of lower-triangle parameters.

    Parameters
    ----------
    C : Any
        MO coefficients.
    space : RotationSpace
        Space definition for packing/unpacking.
    kappa_vec : Any
        Packed parameter vector.
    method : str, optional
        Rotation method (only "cayley" supported).

    Returns
    -------
    Any
        Rotated MO coefficients.
    """

    method_s = str(method).strip().lower()
    xp = _get_xp(C)

    A = space.unpack_to_antisym(kappa_vec)
    if xp is not np:
        A = xp.asarray(A, dtype=xp.float64)

    if method_s == "cayley":
        U = cayley_update(xp, A)
    else:
        raise ValueError("only method='cayley' is supported")

    return C @ U


def reoptimize_casscf(
    scf_out,
    *,
    prev: CASSCFResult,
    mo_coeff: Any | None = None,
    ci0: Any | None = None,
    rotations: Sequence[tuple[int, int, float]] | None = None,
    rotation_mask: np.ndarray | None = None,
    **casscf_kwargs,
) -> CASSCFResult:
    """Restart CASSCF from a previous result, optionally after rotations.

    Parameters
    ----------
    scf_out : Any
        Output of `asuka.frontend.run_*` SCF.
    prev : CASSCFResult
        Previous CASSCFResult (provides ncore/ncas/nelecas defaults).
    mo_coeff : Any | None, optional
        If provided, use as initial orbitals (overrides prev.mo_coeff).
    ci0 : Any | None, optional
        Optional initial CI guess. Default None (let the solver initialize).
    rotations : Sequence[tuple[int, int, float]] | None, optional
        Optional pair rotations (i,j,theta_deg). Applied to the initial orbitals.
    rotation_mask : np.ndarray | None, optional
        Optional boolean mask controlling allowed orbital rotations in the
        internal CASSCF orbital optimizer.
    **casscf_kwargs : Any
        Forwarded to `run_casscf(...)`.

    Returns
    -------
    CASSCFResult
        New CASSCF result.
    """

    C0 = prev.mo_coeff if mo_coeff is None else mo_coeff
    if rotations:
        C0 = rotate_pairs(C0, rotations, degrees=True)

    return run_casscf(
        scf_out,
        ncore=int(prev.ncore),
        ncas=int(prev.ncas),
        nelecas=prev.nelecas,
        mo_coeff0=C0,
        ci0=ci0,
        rotation_mask=rotation_mask,
        **casscf_kwargs,
    )


__all__ = ["RotationSpace", "reoptimize_casscf", "rotate_kappa_vec", "rotate_pairs"]

