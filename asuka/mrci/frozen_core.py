from __future__ import annotations

from dataclasses import dataclass

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
