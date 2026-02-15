from __future__ import annotations

import numpy as np

from asuka.integrals.df_integrals import DFMOIntegrals
from asuka.cuguga.drt import DRT
from asuka.cuguga.oracle import _STEP_TO_OCC, _assign_spin_occupations


def diagonal_element_det_guess_df(drt: DRT, h1e, df_eri: DFMOIntegrals, j: int) -> float:
    """Determinant-based diagonal guess for CSF j using DF/Cholesky vectors.

    Mirrors :func:`asuka.cuguga.oracle.diagonal_element_det_guess`, but
    avoids materializing the dense ERI tensor by evaluating the Coulomb and
    exchange terms directly from the DF factorization.
    """

    j = int(j)
    norb = int(drt.norb)
    if int(df_eri.norb) != norb:
        raise ValueError(f"df_eri.norb={int(df_eri.norb)} does not match drt.norb={norb}")

    h1e = np.asarray(h1e, dtype=np.float64)
    if h1e.shape != (norb, norb):
        raise ValueError("h1e has wrong shape")

    steps = drt.index_to_path(j).astype(np.int8, copy=False)
    occ = _STEP_TO_OCC[steps].astype(np.int8, copy=False)
    neleca = (int(drt.nelec) + int(drt.twos_target)) // 2
    nelecb = int(drt.nelec) - neleca
    alpha, beta = _assign_spin_occupations(occ, neleca=neleca, nelecb=nelecb)

    n = (alpha + beta).astype(np.float64)
    alpha_f = alpha.astype(np.float64)
    beta_f = beta.astype(np.float64)

    hdiag = float(np.dot(np.diag(h1e), n))

    # ecoul = 1/2 * sum_{ij} n_i n_j (ii|jj)
    # with (ii|jj) = sum_L d[L,ii] d[L,jj]
    diag_ids = np.arange(norb, dtype=np.int32) * (norb + 1)
    w = n @ df_eri.l_full[diag_ids]  # (naux,)
    ecoul = 0.5 * float(np.dot(w, w))

    def _exchange(occ_spin: np.ndarray) -> float:
        occ_idx = np.nonzero(occ_spin > 0.5)[0].astype(np.int32, copy=False)
        if occ_idx.size == 0:
            return 0.0
        # sum_{ij} occ_i occ_j (ij|ji); in DF with symmetric pair vectors this is
        # sum_{ij,L} d[L,ij]^2 for i,j in occ.
        pair_ids = (occ_idx[:, None] * norb + occ_idx[None, :]).ravel()
        d = df_eri.l_full[pair_ids]
        val = float(np.sum(d * d))
        return -0.5 * val

    exa = _exchange(alpha_f)
    exb = _exchange(beta_f)
    return hdiag + ecoul + exa + exb
