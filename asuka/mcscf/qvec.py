"""Q-vector for orbital-CI coupling in the CASSCF second-order optimizer.

Based on the Qvec implementation in the BAGEL package
(https://github.com/qsimulate-open/bagel, GPLv3+).

Q[r, u] = sum_{t,x,y} (rt|xy) Gamma2[t,u,x,y]

Built via DF:
  1. Half-transform of active MOs:  half_act[mu, t, P] = sum_nu B[mu, nu, P] C_act[nu, t]
  2. Full active transform:         full[t, u, P] = sum_mu C_act[mu, u] half_act[mu, t, P]
  3. Apply JJ (since B is already J^{-1/2}-transformed, full is already J-dressed)
  4. Contract with 2-RDM:           prdm[t, u, P] = sum_{x,y} full[x, y, P] Gamma2[x,y,t,u]
  5. Back-transform:                tmp[mu, u] = sum_{t,P} half_act[mu, t, P] prdm[t, u, P]
  6. MO transform:                  Q[r, u] = sum_mu C[mu, r] tmp[mu, u]
"""

from __future__ import annotations

import numpy as np


def build_qvec(
    nmo: int,
    nact: int,
    coeff: np.ndarray,
    nclosed: int,
    B_ao: np.ndarray,
    rdm2_av: np.ndarray,
) -> np.ndarray:
    """Compute the Q-vector Q[r, u] = sum_{txy} (rt|xy) Gamma2[tu,xy].

    Parameters
    ----------
    nmo : int
        Total number of MOs.
    nact : int
        Number of active orbitals.
    coeff : np.ndarray
        MO coefficients (nao, nmo).
    nclosed : int
        Number of closed orbitals.
    B_ao : np.ndarray
        Whitened DF factors (nao, nao, naux).  Since these are already
        J^{-1/2}-transformed, (B B^T) gives ERIs directly.
    rdm2_av : np.ndarray
        State-averaged 2-RDM (nact, nact, nact, nact).

    Returns
    -------
    Q : np.ndarray
        Qvec matrix (nmo, nact).
    """
    nao = int(coeff.shape[0])
    naux = int(B_ao.shape[2])
    nocc = nclosed + nact

    C_act = coeff[:, nclosed:nocc]  # (nao, nact)

    # Use the array module of B_ao (cupy or numpy)
    try:
        import cupy as _cp
        _xp = _cp if isinstance(B_ao, _cp.ndarray) else np
    except ImportError:
        _xp = np

    # half-transform: half[mu, t, P] = sum_nu B[mu, nu, P] C_act[nu, t]
    half_act = _xp.einsum("mnP,nt->mtP", B_ao, C_act)  # (nao, nact, naux)

    # full transform: full[t, u, P] = sum_mu C_act[mu, u] half_act[mu, t, P]
    full_act = _xp.einsum("mu,mtP->tuP", C_act, half_act)  # (nact, nact, naux)

    # Since B_ao is already J^{-1/2}-whitened, full_act is already
    # effectively (t u | P) with the Coulomb metric applied once.
    # Apply J twice to the full active integrals.
    # With ASUKA's whitened B, one contraction with B already includes J^{-1/2},
    # and a second contraction includes another J^{-1/2}, giving J^{-1} total.
    # So full_act = sum_P L[t,P] L[u,P] where L is the half-transformed
    # Cholesky vector.  No further J application needed.

    # Contract with 2-RDM:  prdm[t, u, P] = sum_{x,y} full[x, y, P] Gamma2[x, y, t, u]
    rdm2_dev = _xp.asarray(rdm2_av, dtype=_xp.float64)
    rdm2_flat = rdm2_dev.reshape(nact * nact, nact * nact)
    full_flat = full_act.reshape(nact * nact, naux)
    prdm_flat = rdm2_flat.T @ full_flat  # (nact^2, naux)
    prdm = prdm_flat.reshape(nact, nact, naux)

    # Back-transform: tmp[mu, u] = sum_{t,P} half_act[mu, t, P] prdm[t, u, P]
    tmp = _xp.einsum("mtP,tuP->mu", half_act, prdm)  # (nao, nact)

    # MO transform: Q[r, u] = sum_mu C[mu, r] tmp[mu, u]
    Q = coeff.T @ tmp  # (nmo, nact)

    return Q
