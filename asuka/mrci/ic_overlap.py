from __future__ import annotations

import numpy as np

from asuka.mrci.ic_basis import ICDoubles, ICSingles


def apply_overlap_ref_singles(
    *,
    c0: float,
    c_singles: np.ndarray,
    singles: ICSingles,
    gamma: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Apply the overlap metric S to [reference + singles] coefficients.

    Computes ``rho = S @ c`` for the reference and singles block, assuming
    external orbitals are empty in the reference.

    Parameters
    ----------
    c0 : float
        Reference coefficient.
    c_singles : np.ndarray
        Singles coefficients. Shape: (n_singles,).
    singles : ICSingles
        Singles label set.
    gamma : np.ndarray
        1-RDM on internal orbitals. Shape: (nI, nI).

    Returns
    -------
    rho0 : float
        Overlap-transformed reference coefficient (rho0 = c0).
    rho_singles : np.ndarray
        Overlap-transformed singles coefficients. Shape: (n_singles,).
    """

    gamma = np.asarray(gamma, dtype=np.float64)
    if gamma.ndim != 2 or gamma.shape[0] != gamma.shape[1]:
        raise ValueError("gamma must be square")

    c_singles = np.asarray(c_singles, dtype=np.float64).ravel()
    if int(c_singles.size) != int(singles.nlab):
        raise ValueError("c_singles has wrong length for singles label set")

    if int(singles.nlab) == 0:
        return float(c0), np.zeros(0, dtype=np.float64)

    # Assume internal orbitals are indexed densely 0..nI-1 or are already mapped.
    # For the first pass, we require that gamma is indexed by the *global* orbital ids
    # used in singles.r.
    r = np.asarray(singles.r, dtype=np.int64)
    if np.any(r < 0) or np.any(r >= int(gamma.shape[0])):
        raise ValueError("singles.r out of range for gamma")

    rho = np.zeros_like(c_singles)

    order = np.asarray(singles.a_group_order, dtype=np.int32)
    offsets = np.asarray(singles.a_group_offsets, dtype=np.int32)

    # Scratch vector for one external index group (sized to gamma dimension).
    nI = int(gamma.shape[0])
    x = np.zeros(nI, dtype=np.float64)

    for g in range(int(singles.n_groups)):
        start = int(offsets[g])
        stop = int(offsets[g + 1])
        if start == stop:
            continue
        idx = order[start:stop].astype(np.int64, copy=False)
        r_idx = r[idx]

        x.fill(0.0)
        x[r_idx] = c_singles[idx]
        y = gamma @ x
        rho[idx] = y[r_idx]

    return float(c0), np.asarray(rho, dtype=np.float64)


def apply_overlap_doubles(
    *,
    c_doubles: np.ndarray,
    doubles: ICDoubles,
    dm2: np.ndarray,
) -> np.ndarray:
    """Apply the doubles-doubles overlap metric S_dd to doubles coefficients.

    Computes ``rho_d = S_dd @ c_d``, assuming external orbitals are empty in the reference.
    The overlap is block-diagonal in the external pair (a,b).

    Parameters
    ----------
    c_doubles : np.ndarray
        Doubles coefficients. Shape: (n_doubles,).
    doubles : ICDoubles
        Doubles label set.
    dm2 : np.ndarray
        2-RDM on internal orbitals. Shape: (nI, nI, nI, nI).
        Convention: ``dm2[p,q,r,s] = <E_{pq} E_{rs}> - delta_{qr} <E_{ps}>``.

    Returns
    -------
    rho_doubles : np.ndarray
        Overlap-transformed doubles coefficients. Shape: (n_doubles,).
    """

    dm2 = np.asarray(dm2, dtype=np.float64)
    if dm2.ndim != 4 or dm2.shape[0] != dm2.shape[1] or dm2.shape[0] != dm2.shape[2] or dm2.shape[0] != dm2.shape[3]:
        raise ValueError("dm2 must have shape (nI, nI, nI, nI)")
    nI = int(dm2.shape[0])

    c_doubles = np.asarray(c_doubles, dtype=np.float64).ravel()
    if int(c_doubles.size) != int(doubles.nlab):
        raise ValueError("c_doubles has wrong length for doubles label set")
    if int(doubles.nlab) == 0:
        return np.zeros(0, dtype=np.float64)

    r = np.asarray(doubles.r, dtype=np.int64)
    s = np.asarray(doubles.s, dtype=np.int64)
    if np.any(r < 0) or np.any(r >= nI) or np.any(s < 0) or np.any(s >= nI):
        raise ValueError("doubles internal indices out of range for dm2")

    # Flatten dm2 into a matrix A[(r,s),(t,u)] = dm2[r,t,s,u].
    a_mat = dm2.transpose(0, 2, 1, 3).reshape(nI * nI, nI * nI)

    rho = np.zeros_like(c_doubles)

    order = np.asarray(doubles.ab_group_order, dtype=np.int32)
    offsets = np.asarray(doubles.ab_group_offsets, dtype=np.int32)
    keys = np.asarray(doubles.ab_group_keys, dtype=np.int32)

    x = np.zeros(nI * nI, dtype=np.float64)

    for g in range(int(doubles.n_groups)):
        start = int(offsets[g])
        stop = int(offsets[g + 1])
        if start == stop:
            continue
        idx = order[start:stop].astype(np.int64, copy=False)
        r_idx = r[idx]
        s_idx = s[idx]

        x.fill(0.0)
        x[r_idx * nI + s_idx] = c_doubles[idx]
        y = a_mat @ x

        flat = r_idx * nI + s_idx
        if int(keys[g, 0]) == int(keys[g, 1]):
            flat_swapped = s_idx * nI + r_idx
            rho[idx] = y[flat] + y[flat_swapped]
        else:
            rho[idx] = y[flat]

    return np.asarray(rho, dtype=np.float64)


def apply_overlap_ref_singles_doubles(
    *,
    c0: float,
    c_singles: np.ndarray,
    c_doubles: np.ndarray,
    singles: ICSingles,
    doubles: ICDoubles,
    gamma: np.ndarray,
    dm2: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Apply the overlap metric S to [reference + singles + doubles] coefficients.

    Cross blocks vanish under the "external orbitals empty in the reference" assumption.

    Parameters
    ----------
    c0 : float
        Reference coefficient.
    c_singles : np.ndarray
        Singles coefficients.
    c_doubles : np.ndarray
        Doubles coefficients.
    singles : ICSingles
        Singles label set.
    doubles : ICDoubles
        Doubles label set.
    gamma : np.ndarray
        1-RDM on internal orbitals.
    dm2 : np.ndarray
        2-RDM on internal orbitals.

    Returns
    -------
    rho0 : float
        Overlap-transformed reference coefficient.
    rho_s : np.ndarray
        Overlap-transformed singles coefficients.
    rho_d : np.ndarray
        Overlap-transformed doubles coefficients.
    """

    rho0, rho_s = apply_overlap_ref_singles(c0=c0, c_singles=c_singles, singles=singles, gamma=gamma)
    rho_d = apply_overlap_doubles(c_doubles=c_doubles, doubles=doubles, dm2=dm2)
    return rho0, rho_s, rho_d
