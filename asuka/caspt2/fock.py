"""Fock matrix construction for CASPT2.

Ports OpenMolcas ``fmat_caspt2.f`` and ``newfock.f``.
Builds inactive Fock, active Fock, and full state-specific Fock in the MO basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CASPT2Fock:
    """Fock matrices in MO basis for CASPT2."""

    fimo: np.ndarray   # (nmo, nmo) inactive Fock: h + J_core - 0.5*K_core
    famo: np.ndarray   # (nmo, nmo) active Fock: J_act - 0.5*K_act
    fifa: np.ndarray   # (nmo, nmo) full Fock = fimo + famo
    epsa: np.ndarray   # (nash,) diagonal active Fock eigenvalues
    e_core: float      # core energy: Tr[h * D_core] + 0.5*Tr[V_core * D_core] + E_nuc


def build_caspt2_fock(
    h1e_mo: np.ndarray,
    eri_mo: np.ndarray,
    dm1_act: np.ndarray,
    nish: int,
    nash: int,
    nssh: int,
    *,
    e_nuc: float = 0.0,
) -> CASPT2Fock:
    """Build CASPT2 Fock matrices from MO integrals and active 1-RDM.

    Parameters
    ----------
    h1e_mo : (nmo, nmo)
        Core Hamiltonian in MO basis.
    eri_mo : (nmo, nmo, nmo, nmo)
        Two-electron integrals in MO basis, chemists' notation (pq|rs).
    dm1_act : (nash, nash)
        Active-space 1-RDM.
    nish, nash, nssh : int
        Number of inactive, active, secondary orbitals.
    e_nuc : float
        Nuclear repulsion energy.
    """
    nmo = nish + nash + nssh
    h1e_mo = np.asarray(h1e_mo, dtype=np.float64)
    eri_mo = np.asarray(eri_mo, dtype=np.float64)
    dm1_act = np.asarray(dm1_act, dtype=np.float64)

    if h1e_mo.shape != (nmo, nmo):
        raise ValueError(f"h1e_mo shape {h1e_mo.shape} != ({nmo}, {nmo})")
    if eri_mo.shape != (nmo, nmo, nmo, nmo):
        raise ValueError(f"eri_mo shape {eri_mo.shape} != ({nmo},)*4")
    if dm1_act.shape != (nash, nash):
        raise ValueError(f"dm1_act shape {dm1_act.shape} != ({nash}, {nash})")

    act = slice(nish, nish + nash)

    # -- Inactive Fock --
    # F^I_pq = h_pq + sum_i [2*(pq|ii) - (pi|iq)]
    fimo = h1e_mo.copy()
    for i in range(nish):
        fimo += 2.0 * eri_mo[:, :, i, i] - eri_mo[:, i, i, :]

    # -- Active Fock --
    # OpenMolcas (`fmat_caspt2.f`) uses:
    #   F^A(p,q) = (1/2) * sum_{t,u} D(t,u) * ( 2*(pq|tu) - (pu|qt) )
    # For symmetric D(t,u) this is equivalent to the standard spin-free form:
    #   F^A(p,q) = sum_{t,u} D(t,u) * ( (pq|tu) - 0.5*(pt|qu) )
    famo = np.zeros((nmo, nmo), dtype=np.float64)
    for t in range(nash):
        for u in range(nash):
            d_tu = dm1_act[t, u]
            if abs(d_tu) < 1e-15:
                continue
            tt = t + nish
            uu = u + nish
            famo += d_tu * (eri_mo[:, :, tt, uu] - 0.5 * eri_mo[:, tt, :, uu])

    # -- Full Fock --
    fifa = fimo + famo

    # -- Active Fock eigenvalues (diagonal of F^{I+A} in active block) --
    epsa = np.diag(fifa[act, act]).copy()

    # -- Core energy --
    # E_core = sum_i [h_ii + F^I_ii] + E_nuc
    #        = Tr[h * D_core] + 0.5*Tr[V_core * D_core] + E_nuc
    e_core = float(e_nuc)
    for i in range(nish):
        e_core += float(h1e_mo[i, i] + fimo[i, i])

    return CASPT2Fock(
        fimo=np.asarray(fimo, dtype=np.float64, order="C"),
        famo=np.asarray(famo, dtype=np.float64, order="C"),
        fifa=np.asarray(fifa, dtype=np.float64, order="C"),
        epsa=np.asarray(epsa, dtype=np.float64),
        e_core=e_core,
    )
