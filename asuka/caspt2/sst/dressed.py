"""Dressed-orbital utilities for SST.

In the SST formulation, the MP2-like contribution uses *dressed* orbital
energies: diagonal elements (or semicanonical eigenvalues) of the full
state-specific CASPT2 Fock matrix.

ASUKA's internally-contracted CASPT2 implementation already uses the diagonal
of ``fock.fifa`` as orbital energies in the denominators, and retains the
full ``fock.fifa`` for sigma-coupling when the orbitals are not perfectly
semicanonical.

For the SST backend, the MP2-like H± sector is evaluated using the same
convention (diagonal of ``fifa``).  This module provides convenience helpers
for:

  * extracting orbital energies for core/active/virtual blocks
  * (optional) semicanonicalizing core and virtual subspaces for diagnostics
    or for future reduced-scaling SST variants

The current SST backend does **not** require semicanonicalization to run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asuka.caspt2.fock import CASPT2Fock
from asuka.mrpt2.semicanonical import SemicanonicalCoreVirt, molcas_diafck_eigh

__all__ = [
    "DressedEnergies",
    "extract_dressed_energies",
    "semicanonicalize_core_virt_from_mo_fock",
]


@dataclass(frozen=True)
class DressedEnergies:
    """Blockwise orbital energies used in SST denominators."""

    eps_core: np.ndarray  # (ncore,)
    eps_act: np.ndarray   # (ncas,)
    eps_virt: np.ndarray  # (nvirt,)


def extract_dressed_energies(
    fock: CASPT2Fock,
    *,
    ncore: int,
    ncas: int,
    nvirt: int,
) -> DressedEnergies:
    """Extract the diagonal orbital energies from the MO Fock ``fifa``.

    Parameters
    ----------
    fock
        CASPT2 Fock object.
    ncore, ncas, nvirt
        Orbital partition sizes.

    Returns
    -------
    DressedEnergies
        ``eps_core``, ``eps_act``, ``eps_virt`` extracted from the diagonal of
        ``fock.fifa``.
    """
    fifa = np.asarray(fock.fifa, dtype=np.float64)
    if fifa.ndim != 2 or fifa.shape[0] != fifa.shape[1]:
        raise ValueError("fock.fifa must be a square 2D array")

    ncore = int(ncore)
    ncas = int(ncas)
    nvirt = int(nvirt)
    nmo = int(fifa.shape[0])

    if ncore < 0 or ncas < 0 or nvirt < 0:
        raise ValueError("invalid orbital partition sizes")
    if ncore + ncas + nvirt != nmo:
        raise ValueError("ncore+ncas+nvirt must equal nmo")

    occ = ncore + ncas

    eps_core = np.diag(fifa[:ncore, :ncore]).copy() if ncore > 0 else np.zeros((0,), dtype=np.float64)
    eps_act = np.diag(fifa[ncore:occ, ncore:occ]).copy() if ncas > 0 else np.zeros((0,), dtype=np.float64)
    eps_virt = np.diag(fifa[occ:occ + nvirt, occ:occ + nvirt]).copy() if nvirt > 0 else np.zeros((0,), dtype=np.float64)

    return DressedEnergies(
        eps_core=np.asarray(eps_core, dtype=np.float64, order="C"),
        eps_act=np.asarray(eps_act, dtype=np.float64, order="C"),
        eps_virt=np.asarray(eps_virt, dtype=np.float64, order="C"),
    )


def semicanonicalize_core_virt_from_mo_fock(
    fock: CASPT2Fock,
    mo_coeff: np.ndarray,
    *,
    ncore: int,
    ncas: int,
    nvirt: int,
    deg_tol: float = 1e-10,
) -> SemicanonicalCoreVirt:
    """Semicanonicalize the core and virtual subspaces using the MO Fock.

    This mirrors the idea of diagonalizing the core-core and virtual-virtual
    blocks of the generalized Fock, but performs it directly in the current
    MO basis using ``fock.fifa``.

    The resulting rotations can be used to rotate orbitals and/or energies.

    Parameters
    ----------
    fock
        CASPT2 Fock object.
    mo_coeff
        Full MO coefficient matrix in AO basis (nao, nmo).
    ncore, ncas, nvirt
        Orbital partition sizes.
    deg_tol
        Degeneracy tolerance passed to the Molcas-compatible diagonalizer.

    Returns
    -------
    SemicanonicalCoreVirt
        Semicanonicalized core/virtual AO orbitals and eigenvalues.
    """
    fifa = np.asarray(fock.fifa, dtype=np.float64)
    C = np.asarray(mo_coeff, dtype=np.float64)
    if fifa.ndim != 2 or fifa.shape[0] != fifa.shape[1]:
        raise ValueError("fock.fifa must be a square 2D array")
    nmo = int(fifa.shape[0])
    if C.ndim != 2 or int(C.shape[1]) != nmo:
        raise ValueError("mo_coeff shape mismatch with fock.fifa")

    ncore = int(ncore)
    ncas = int(ncas)
    nvirt = int(nvirt)
    if ncore + ncas + nvirt != nmo:
        raise ValueError("ncore+ncas+nvirt must equal nmo")

    nocc = ncore + ncas

    # Core block
    if ncore > 0:
        f_cc = 0.5 * (fifa[:ncore, :ncore] + fifa[:ncore, :ncore].T)
        eps_core, u_core = molcas_diafck_eigh(f_cc, deg_tol=float(deg_tol))
        mo_core_sc = C[:, :ncore] @ u_core
    else:
        eps_core = np.zeros((0,), dtype=np.float64)
        u_core = np.zeros((0, 0), dtype=np.float64)
        mo_core_sc = C[:, :0]

    # Virtual block
    if nvirt > 0:
        f_vv = fifa[nocc:nocc + nvirt, nocc:nocc + nvirt]
        f_vv = 0.5 * (f_vv + f_vv.T)
        eps_virt, u_virt = molcas_diafck_eigh(f_vv, deg_tol=float(deg_tol))
        mo_virt_sc = C[:, nocc:nocc + nvirt] @ u_virt
    else:
        eps_virt = np.zeros((0,), dtype=np.float64)
        u_virt = np.zeros((0, 0), dtype=np.float64)
        mo_virt_sc = C[:, nocc:nocc]

    return SemicanonicalCoreVirt(
        mo_core=np.asarray(mo_core_sc, dtype=np.float64, order="C"),
        mo_virt=np.asarray(mo_virt_sc, dtype=np.float64, order="C"),
        eps_core=np.asarray(eps_core, dtype=np.float64, order="C"),
        eps_virt=np.asarray(eps_virt, dtype=np.float64, order="C"),
        u_core=np.asarray(u_core, dtype=np.float64, order="C"),
        u_virt=np.asarray(u_virt, dtype=np.float64, order="C"),
    )
