"""State-Lagrangian helpers for MS/XMS-CASPT2 gradients.

This module implements the explicit state-weight part of the OpenMolcas
``SLag`` construction used by multistate CASPT2 gradients.
"""

from __future__ import annotations

import numpy as np


def build_state_lagrangian(
    nstates: int,
    ueff: np.ndarray,
    iroot: int,
    *,
    is_xms: bool,
    u0: np.ndarray | None = None,
) -> np.ndarray:
    """Build the explicit multistate ``SLag`` contribution for one target root.

    This matches the state-weight construction in OpenMolcas
    ``caspt2_grad.f`` after ``CLagFinal``:

    - MS gradients only add diagonal target-state weights from ``UEFF``.
    - XMS gradients add the lower-triangular state-rotation weights, with a
      factor of 2 on off-diagonal entries.
    - In both cases, the original SA-CASSCF target-root contribution is
      removed by subtracting 1 on the selected target diagonal.

    Parameters
    ----------
    nstates
        Number of model-space states.
    ueff
        Effective-Hamiltonian eigenvectors, shape ``(nstates, nstates)``.
    iroot
        Target root index in the final MS/XMS basis.
    is_xms
        Whether to build the XMS lower-triangular form.
    u0
        Optional XMS reference-rotation matrix. Accepted for API symmetry and
        validation hooks; not needed for the explicit ``UEFF``-weight term.
    """

    nstates_i = int(nstates)
    if nstates_i < 1:
        raise ValueError("nstates must be >= 1")

    u = np.asarray(ueff, dtype=np.float64)
    if u.shape != (nstates_i, nstates_i):
        raise ValueError(
            f"ueff shape mismatch: expected {(nstates_i, nstates_i)}, got {u.shape}"
        )

    iroot_i = int(iroot)
    if iroot_i < 0 or iroot_i >= nstates_i:
        raise ValueError(f"iroot out of range: {iroot_i} not in [0, {nstates_i})")

    if u0 is not None:
        u0_a = np.asarray(u0, dtype=np.float64)
        if u0_a.shape != (nstates_i, nstates_i):
            raise ValueError(
                f"u0 shape mismatch: expected {(nstates_i, nstates_i)}, got {u0_a.shape}"
            )

    slag = np.zeros((nstates_i, nstates_i), dtype=np.float64)
    if bool(is_xms):
        for ist in range(nstates_i):
            for jst in range(ist + 1):
                val = float(u[ist, iroot_i] * u[jst, iroot_i])
                if ist != jst:
                    val *= 2.0
                slag[ist, jst] += val
    else:
        for ist in range(nstates_i):
            slag[ist, ist] += float(u[ist, iroot_i] * u[ist, iroot_i])

    slag[iroot_i, iroot_i] -= 1.0
    return np.asarray(slag, dtype=np.float64)


__all__ = ["build_state_lagrangian"]
