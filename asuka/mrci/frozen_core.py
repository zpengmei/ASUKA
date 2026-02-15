from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrozenCoreMOIntegrals:
    """Dense MO-basis Hamiltonian payload for a frozen-core correlated subspace."""

    h1e: np.ndarray  # (norb, norb)
    eri4: np.ndarray  # (norb, norb, norb, norb) in chemist notation (pq|rs)
    ecore: float


def frozen_core_from_eri4(
    *,
    h1e: np.ndarray,
    eri4: np.ndarray,
    ncore: int,
    e_nuc: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (h1e_corr, eri4_corr, ecore) for orbitals ordered as [core][corr].

    Notes
    -----
    This is a pure-Numpy helper that assumes `h1e` and `eri4` are in a common MO basis
    for *all* orbitals (core + correlated). The correlated orbitals are assumed to be
    the trailing block `[ncore:]`.
    """

    h1e = np.asarray(h1e, dtype=np.float64)
    eri4 = np.asarray(eri4, dtype=np.float64)
    ncore = int(ncore)
    e_nuc = float(e_nuc)

    if h1e.ndim != 2 or h1e.shape[0] != h1e.shape[1]:
        raise ValueError("h1e must be square")
    nmo = int(h1e.shape[0])
    if eri4.shape != (nmo, nmo, nmo, nmo):
        raise ValueError("eri4 has wrong shape")
    if ncore < 0 or ncore > nmo:
        raise ValueError("ncore must satisfy 0 <= ncore <= nmo")

    if ncore == 0:
        return h1e.copy(), eri4.copy(), float(e_nuc)
    if ncore == nmo:
        e1 = 2.0 * float(np.trace(h1e))
        e2 = 2.0 * float(np.einsum("iijj->", eri4)) - float(np.einsum("ijji->", eri4))
        return (
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0, 0, 0, 0), dtype=np.float64),
            float(e_nuc + e1 + e2),
        )

    core = np.arange(ncore, dtype=np.intp)
    corr = np.arange(ncore, nmo, dtype=np.intp)

    h_cc = h1e[np.ix_(corr, corr)].copy()
    eri_corr = eri4[np.ix_(corr, corr, corr, corr)].copy()

    # h_eff[p,q] = h[p,q] + Σ_i(core) [ 2 (p q| i i) - (p i| i q) ].
    eri_pqii = eri4[np.ix_(corr, corr, core, core)]  # (p q| i j)
    eri_piiq = eri4[np.ix_(corr, core, core, corr)]  # (p i| j q)
    vcore = 2.0 * np.einsum("pqii->pq", eri_pqii, optimize=True) - np.einsum("piiq->pq", eri_piiq, optimize=True)
    h_cc += vcore

    # E_core = E_nuc + 2 Σ_i h[i,i] + Σ_{i,j} [ 2 (i i| j j) - (i j| j i) ].
    h_ii = np.trace(h1e[np.ix_(core, core)])
    eri_core = eri4[np.ix_(core, core, core, core)]
    e_coul = float(np.einsum("iijj->", eri_core, optimize=True))
    e_ex = float(np.einsum("ijji->", eri_core, optimize=True))
    ecore = float(e_nuc + 2.0 * float(h_ii) + (2.0 * e_coul - e_ex))

    return np.asarray(h_cc, dtype=np.float64, order="C"), np.asarray(eri_corr, dtype=np.float64, order="C"), ecore


def _frozen_core_h1e_ecore_pyscf(
    *,
    mol,
    mf_or_mc: Any,
    mo_core: np.ndarray,
    mo_corr: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return (h1e_corr, ecore) using PySCF AO potentials for the frozen-core density.

    This is a PySCF-compatibility helper and is intentionally treated as an internal API.
    """

    mo_core = np.asarray(mo_core, dtype=np.float64)
    mo_corr = np.asarray(mo_corr, dtype=np.float64)
    if mo_core.ndim != 2 or mo_corr.ndim != 2:
        raise ValueError("mo_core and mo_corr must be 2D arrays")
    if mo_core.shape[0] != mo_corr.shape[0]:
        raise ValueError("mo_core and mo_corr must have the same number of rows (AO dimension)")

    ncore = int(mo_core.shape[1])
    e_nuc = float(getattr(mol, "energy_nuc")())

    if ncore == 0:
        hcore = np.asarray(mf_or_mc.get_hcore(mol), dtype=np.float64)
        h1e_corr = mo_corr.T @ hcore @ mo_corr
        return np.asarray(h1e_corr, dtype=np.float64, order="C"), e_nuc

    dm_core = 2.0 * (mo_core @ mo_core.T)
    hcore = np.asarray(mf_or_mc.get_hcore(mol), dtype=np.float64)
    vhf_core = np.asarray(mf_or_mc.get_veff(mol, dm_core), dtype=np.float64)

    h_eff_ao = hcore + vhf_core
    h1e_corr = mo_corr.T @ h_eff_ao @ mo_corr

    e1 = float(np.einsum("ij,ji->", dm_core, hcore, optimize=True))
    e2 = 0.5 * float(np.einsum("ij,ji->", dm_core, vhf_core, optimize=True))
    ecore = float(e_nuc + e1 + e2)

    return np.asarray(h1e_corr, dtype=np.float64, order="C"), ecore


def _build_frozen_core_mo_integrals_pyscf(
    *,
    mol,
    mf_or_mc: Any,
    mo_core: np.ndarray,
    mo_corr: np.ndarray,
) -> FrozenCoreMOIntegrals:
    """Build dense MO integrals for correlated orbitals with a frozen-core shift.

    This is a PySCF-compatibility helper and is intentionally treated as an internal API.
    """

    from asuka.cueri.ao2mo import ao2mo_kernel  # noqa: PLC0415

    h1e_corr, ecore = _frozen_core_h1e_ecore_pyscf(mol=mol, mf_or_mc=mf_or_mc, mo_core=mo_core, mo_corr=mo_corr)

    norb = int(np.asarray(mo_corr).shape[1])
    eri_mat = ao2mo_kernel(mol, np.asarray(mo_corr, dtype=np.float64), compact=False)
    eri4_c = np.asarray(eri_mat.reshape(norb, norb, norb, norb), dtype=np.float64, order="C")

    return FrozenCoreMOIntegrals(h1e=h1e_corr, eri4=eri4_c, ecore=float(ecore))
