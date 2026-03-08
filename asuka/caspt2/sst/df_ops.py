"""Density-fitting helper utilities for the SST backend.

The SST backend is designed so that (eventually) the expensive pieces can be
expressed in terms of MP2-like contractions and Fock builds.

For the first implementation stage, we keep things simple and provide
CPU/Numpy utilities that:

  * build the 3-index DF tensor in the MO basis, ``B_mo[p,q,Q]``
  * build the full 4-index MO ERIs from DF factors (debug/validation only)
  * build the core-virtual DF pair block ``L_ai[a,i,Q]`` used by the MP2-like
    H± energy

These helpers are intentionally small, dependency-free, and easy to unit test.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "build_B_mo_from_B_ao",
    "build_full_mo_eris_from_B_ao",
    "build_L_ai_from_B_ao",
]


def _as_f64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def build_B_mo_from_B_ao(B_ao: np.ndarray, mo_coeff: np.ndarray) -> np.ndarray:
    """Transform AO DF factors to MO DF factors.

    Parameters
    ----------
    B_ao
        AO DF factors with shape (nao, nao, naux). In typical DF/RI notation,
        these are the 3-index quantities such that:

            (mu nu | lambda sigma) = sum_Q B_ao[mu,nu,Q] * B_ao[lambda,sigma,Q]

        Many DF builders provide symmetric (mu,nu) blocks, but symmetry is not
        assumed here.
    mo_coeff
        MO coefficient matrix (nao, nmo).

    Returns
    -------
    B_mo
        MO DF factors (nmo, nmo, naux) where:

            B_mo[p,q,Q] = sum_{mu,nu} C[mu,p] * B_ao[mu,nu,Q] * C[nu,q]
    """
    B_ao = _as_f64(B_ao)
    C = _as_f64(mo_coeff)
    if B_ao.ndim != 3:
        raise ValueError("B_ao must be 3D (nao,nao,naux)")
    if C.ndim != 2:
        raise ValueError("mo_coeff must be 2D (nao,nmo)")

    nao, nao2, naux = map(int, B_ao.shape)
    if nao2 != nao:
        raise ValueError("B_ao must have shape (nao,nao,naux)")
    if int(C.shape[0]) != nao:
        raise ValueError("mo_coeff AO dimension mismatch")
    nmo = int(C.shape[1])

    # (nao, nao*naux)
    B_flat = B_ao.reshape(nao, nao * naux)

    # tmp[p, nu, Q] = sum_mu C[mu,p] * B_ao[mu,nu,Q]
    tmp = C.T @ B_flat  # (nmo, nao*naux)
    tmp = tmp.reshape(nmo, nao, naux)

    # B_mo[p,q,Q] = sum_nu tmp[p,nu,Q] * C[nu,q]
    B_mo = np.einsum("pnQ,nq->pqQ", tmp, C, optimize=True)
    return np.asarray(B_mo, dtype=np.float64, order="C")


def build_full_mo_eris_from_B_ao(B_ao: np.ndarray, mo_coeff: np.ndarray) -> np.ndarray:
    """Build full 4-index MO ERIs from DF factors.

    Notes
    -----
    This is **O(nmo^5)** work and **O(nmo^4)** memory. It is intended only for:
      * unit tests
      * CPU reference validation
      * tiny molecules

    For production CASPT2, use DF-native kernels (as in ASUKA's CUDA backend).

    Parameters
    ----------
    B_ao
        AO DF factors (nao, nao, naux).
    mo_coeff
        MO coefficient matrix (nao, nmo).

    Returns
    -------
    eri_mo
        Full ERIs in chemists' notation (nmo, nmo, nmo, nmo):

            (pq|rs) = sum_Q B_mo[p,q,Q] * B_mo[r,s,Q]
    """
    B_mo = build_B_mo_from_B_ao(B_ao, mo_coeff)
    eri_mo = np.einsum("pqQ,rsQ->pqrs", B_mo, B_mo, optimize=True)
    return np.asarray(eri_mo, dtype=np.float64, order="C")


def build_L_ai_from_B_ao(
    B_ao: np.ndarray,
    mo_coeff: np.ndarray,
    *,
    ncore: int,
    ncas: int,
    nvirt: int,
) -> np.ndarray:
    """Build the DF core-virtual pair block L_ai[a,i,Q].

    This is the only DF object required by the MP2-like H± energy evaluation.

    Parameters
    ----------
    B_ao
        AO DF factors (nao, nao, naux).
    mo_coeff
        MO coefficient matrix (nao, nmo).
    ncore, ncas, nvirt
        Orbital partition sizes. Virtual orbitals are assumed to begin at
        index ``ncore + ncas``.

    Returns
    -------
    L_ai
        Array with shape (nvirt, ncore, naux) satisfying:

            L_ai[a,i,Q] = sum_{mu,nu} C[mu,virt(a)] * B_ao[mu,nu,Q] * C[nu,core(i)]

    Notes
    -----
    The current implementation is straightforward and uses a reshape + GEMM +
    einsum sequence. It is meant for CPU reference and small problems.
    """
    B_ao = _as_f64(B_ao)
    C = _as_f64(mo_coeff)

    if int(ncore) < 0 or int(ncas) < 0 or int(nvirt) < 0:
        raise ValueError("invalid orbital counts")

    nao, nao2, naux = map(int, B_ao.shape)
    if nao2 != nao:
        raise ValueError("B_ao must have shape (nao,nao,naux)")
    if int(C.shape[0]) != nao:
        raise ValueError("mo_coeff AO dimension mismatch")

    nocc = int(ncore) + int(ncas)
    if nocc + int(nvirt) > int(C.shape[1]):
        raise ValueError("ncore+ncas+nvirt exceeds mo_coeff nmo")

    if int(ncore) == 0 or int(nvirt) == 0 or int(naux) == 0:
        return np.zeros((int(nvirt), int(ncore), int(naux)), dtype=np.float64)

    C_core = C[:, : int(ncore)]
    C_virt = C[:, nocc : nocc + int(nvirt)]

    # Flatten B_ao as (nao, nao*naux)
    B_flat = B_ao.reshape(nao, nao * naux)

    # tmp[a, nu, Q] = sum_mu C_virt[mu,a] * B_ao[mu,nu,Q]
    tmp = C_virt.T @ B_flat  # (nvirt, nao*naux)
    tmp = tmp.reshape(int(nvirt), nao, naux)

    # L_ai[a,i,Q] = sum_nu tmp[a,nu,Q] * C_core[nu,i]
    L_ai = np.einsum("anQ,ni->aiQ", tmp, C_core, optimize=True)
    return np.asarray(L_ai, dtype=np.float64, order="C")
