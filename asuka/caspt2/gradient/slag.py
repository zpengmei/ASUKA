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

    **MS-CASPT2** (``is_xms=False``): diagonal only.

    - Diagonal: ``SLag[I,I] = U[I,r]^2``
    - Target root: ``SLag[r,r] -= 1`` (removes SA-CASSCF contribution)

    Off-diagonal MS coupling is handled by CLagFinal (MCLR response), not here.
    Matches Molcas ``caspt2_grad.f`` GrdCls (MS branch).

    **XMS-CASPT2** (``is_xms=True``): full lower-triangular.

    - Diagonal: ``SLag[I,I] = U[I,r]^2``
    - Off-diagonal (I>J): ``SLag[I,J] = 2 * U[I,r] * U[J,r]``
    - Target root: ``SLag[r,r] -= 1``

    Off-diagonal entries drive Phase 2 (SIGDER-based Heff coupling derivative).

    Parameters
    ----------
    nstates
        Number of model-space states.
    ueff
        Effective-Hamiltonian eigenvectors, shape ``(nstates, nstates)``.
    iroot
        Target root index in the final MS/XMS basis.
    is_xms
        True for XMS (full lower-triangular SLag), False for MS (diagonal only).
    u0
        Optional XMS reference-rotation matrix. Accepted for API symmetry and
        validation hooks; not used in the SLag formula itself.
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
        # XMS: full lower-triangular with factor 2 on off-diagonals.
        # SLag[I,I] += U[I,r]^2, SLag[I,J] += 2*U[I,r]*U[J,r] for I>J.
        # Off-diagonal entries drive Phase 2 (SIGDER-based Heff coupling derivative).
        for ist in range(nstates_i):
            for jst in range(ist + 1):
                val = float(u[ist, iroot_i] * u[jst, iroot_i])
                if ist != jst:
                    val *= 2.0
                slag[ist, jst] += val
    else:
        # MS: diagonal only from this function.
        # Off-diagonal MS SLag (for Phase 2 d<I|H|Ω_J>/dR contributions) is added
        # separately in ms_grad.py after this call: slag[I,J] = 2*U[I,r]*U[J,r] for I>J.
        # Molcas handles the equivalent via CLagFinal (MCLR response); ASUKA uses Phase 2
        # (SIGDER-based amplitude + OLag response) as an equivalent approach.
        for ist in range(nstates_i):
            slag[ist, ist] += float(u[ist, iroot_i] * u[ist, iroot_i])

    slag[iroot_i, iroot_i] -= 1.0
    return np.asarray(slag, dtype=np.float64)


__all__ = ["build_state_lagrangian"]
