from __future__ import annotations

from math import isqrt

import numpy as np


def npair(norb: int) -> int:
    """Calculate the number of unique pairs for `norb` orbitals.

    Returns the size of the lower triangle of a symmetric `norb x norb` matrix,
    including the diagonal. Formula: `norb * (norb + 1) / 2`.

    Parameters
    ----------
    norb : int
        Number of orbitals.

    Returns
    -------
    int
        Number of packed pairs (npair).
    """

    if norb < 0:
        raise ValueError("norb must be >= 0")
    return norb * (norb + 1) // 2


def pair_id(p: int, q: int) -> int:
    """Calculate the packed pair index for orbitals (p, q).

    Uses the canonical ordering `p >= q`. The formula is `p * (p + 1) // 2 + q`.
    If `p < q`, the indices are swapped before calculation.

    Parameters
    ----------
    p, q : int
        Orbital indices (0-based).

    Returns
    -------
    int
        The linearized index for the pair (p, q).
    """

    if p < 0 or q < 0:
        raise ValueError("p and q must be >= 0")
    if p >= q:
        return p * (p + 1) // 2 + q
    return q * (q + 1) // 2 + p


def unpack_pair_id(pair: int) -> tuple[int, int]:
    """Recover the orbital indices (p, q) from a packed pair index.

    Inverse operation of `pair_id`. Returns the canonical pair such that `p >= q`.

    Parameters
    ----------
    pair : int
        The packed pair index.

    Returns
    -------
    tuple[int, int]
        The indices `(p, q)`.
    """

    if pair < 0:
        raise ValueError("pair must be >= 0")
    # Find p such that p(p+1)/2 <= pair < (p+1)(p+2)/2.
    p = (isqrt(8 * pair + 1) - 1) // 2
    q = pair - p * (p + 1) // 2
    return int(p), int(q)


def _get_xp(*arrays):
    for arr in arrays:
        if arr is None:
            continue
        mod = type(arr).__module__
        if mod.startswith("cupy"):
            import cupy as cp  # local import to keep NumPy-only workflows working

            return cp
    return np


def pair_id_matrix(norb: int, *, xp=None):
    """Generate a matrix of packed pair indices for all (p, q).

    Constructs a symmetric `(norb, norb)` matrix where element `[p, q]` contains
    the packed index corresponding to the pair `{p, q}`.

    Parameters
    ----------
    norb : int
        Number of orbitals.
    xp : module, optional
        Array backend (numpy or cupy). If None, defaults to numpy.

    Returns
    -------
    xp.ndarray
        A 2D array of int32 indices.
    """

    if norb < 0:
        raise ValueError("norb must be >= 0")
    if xp is None:
        xp = np
    p = xp.arange(norb, dtype=xp.int64)[:, None]
    q = xp.arange(norb, dtype=xp.int64)[None, :]
    mx = xp.maximum(p, q)
    mn = xp.minimum(p, q)
    out = mx * (mx + 1) // 2 + mn
    return out.astype(xp.int32, copy=False)


def ordered_to_packed_index(norb: int, *, xp=None):
    """Generate an index array mapping ordered indices to packed pair indices.

    Creates a flattened array of size `norb^2` where the i-th element (corresponding
    to ordered index `pq = p*norb + q`) contains the packed index `pair_id(p, q)`.
    Useful for expanding packed matrices to full symmetric matrices.

    Parameters
    ----------
    norb : int
        Number of orbitals.
    xp : module, optional
        Array backend.

    Returns
    -------
    xp.ndarray
        1D array of int32 indices.
    """

    return pair_id_matrix(norb, xp=xp).reshape((-1,))


def expand_eri_packed_to_ordered(eri_packed, norb: int):
    """Expand a packed ERI matrix to a full ordered-pair matrix.

    Transforms a `(npair, npair)` matrix into a `(norb^2, norb^2)` matrix,
    duplicating elements according to permutation symmetry (pq|rs) = (qp|rs) = ...

    Parameters
    ----------
    eri_packed : array-like
        The packed ERI matrix.
    norb : int
        Number of orbitals.

    Returns
    -------
    array-like
        The expanded ERI matrix with the same backend as input.
    """

    if norb < 0:
        raise ValueError("norb must be >= 0")
    if getattr(eri_packed, "ndim", None) != 2:
        raise ValueError("eri_packed must be a 2D array")
    if eri_packed.shape[0] != eri_packed.shape[1]:
        raise ValueError("eri_packed must be square")
    expected = npair(norb)
    if int(eri_packed.shape[0]) != expected:
        raise ValueError(f"eri_packed has shape {eri_packed.shape}, expected ({expected},{expected}) for norb={norb}")

    xp = _get_xp(eri_packed)
    idx = ordered_to_packed_index(norb, xp=xp)
    return eri_packed[idx[:, None], idx[None, :]]


def j_ps_from_eri_mat(eri_mat, norb: int):
    """Compute the Coulomb-like matrix J[p,s] from the full ERI matrix.

    Calculates `J_{ps} = Σ_q (pq|qs)` by creating a view of the ERI matrix
    as a 4-tensor and contracting indices.

    Parameters
    ----------
    eri_mat : array-like
        Ordered ERI matrix `(norb^2, norb^2)`.
    norb : int
        Number of orbitals.

    Returns
    -------
    array-like
        The J matrix of shape `(norb, norb)`.
    """

    if norb < 0:
        raise ValueError("norb must be >= 0")
    if getattr(eri_mat, "ndim", None) != 2:
        raise ValueError("eri_mat must be a 2D array")
    if eri_mat.shape != (norb * norb, norb * norb):
        raise ValueError(f"eri_mat must have shape ({norb*norb},{norb*norb}), got {eri_mat.shape}")

    xp = _get_xp(eri_mat)
    g = eri_mat.reshape((norb, norb, norb, norb))
    return xp.einsum("pqqs->ps", g)


def j_ps_from_eri_packed(eri_packed, norb: int):
    """Compute `j_ps[p,s]` directly from packed pair ERIs.

    Computes the Coulomb-like contraction without expanding the full ERI tensor.
    Efficiently handles the mapping from `(p, q)` pairs to packed indices.

    Parameters
    ----------
    eri_packed : array-like
        Packed ERI matrix `(npair, npair)`.
    norb : int
        Number of orbitals.

    Returns
    -------
    array-like
        The J matrix of shape `(norb, norb)`.
    """

    if norb < 0:
        raise ValueError("norb must be >= 0")
    if getattr(eri_packed, "ndim", None) != 2:
        raise ValueError("eri_packed must be a 2D array")
    if eri_packed.shape[0] != eri_packed.shape[1]:
        raise ValueError("eri_packed must be square")
    expected = npair(norb)
    if int(eri_packed.shape[0]) != expected:
        raise ValueError(f"eri_packed has shape {eri_packed.shape}, expected ({expected},{expected}) for norb={norb}")

    xp = _get_xp(eri_packed)
    idx = pair_id_matrix(norb, xp=xp)
    out = xp.zeros((norb, norb), dtype=eri_packed.dtype)
    for q in range(norb):
        out += eri_packed[idx[:, q][:, None], idx[q, :][None, :]]
    return out


def build_pair_coeff_ordered(CA, CB, *, same_shell: bool):
    """Build ordered pair coefficients K_AB for a shell pair (A, B).

    Computes the transformation matrix from AO products `(μν)` to MO pairs `(pq)`.
    Returned shape is `(nA*nB, norb^2)`.
    Formula: `K_{μν, pq} = C_{μp} C_{νq}` (plus `C_{μq} C_{νp}` if `same_shell=False`
    and A != B, to account for swapping).

    Parameters
    ----------
    CA, CB : array-like
        MO coefficients for shells A and B. Shape `(n_func, norb)`.
    same_shell : bool
        True if shell A == shell B. Prevents double-counting when swapping indices.

    Returns
    -------
    array-like
        The pair coefficient matrix.
    """

    xp = _get_xp(CA, CB)
    CA = xp.asarray(CA)
    CB = xp.asarray(CB)
    if CA.ndim != 2 or CB.ndim != 2:
        raise ValueError("CA/CB must be 2D arrays with shape (nAO_in_shell, norb)")
    if CA.shape[1] != CB.shape[1]:
        raise ValueError("CA and CB must have the same norb (2nd dimension)")

    nA, norb = map(int, CA.shape)
    nB = int(CB.shape[0])
    K = xp.einsum("ap,bq->abpq", CA, CB)
    if not same_shell:
        K = K + xp.einsum("bp,aq->abpq", CB, CA)
    return K.reshape((nA * nB, norb * norb))


def build_pair_coeff_ordered_mixed(CA_p, CB_p, CA_u, CB_u, *, same_shell: bool):
    """Build ordered mixed-orbital pair coefficients.

    Constructs coefficients for a mixed MO pair space `(p, u)`, mapping AO products
    `(μν)` to `(pu)`. Used for generalized gradients or active-space terms requiring
    pairs between different MO sets.

    Parameters
    ----------
    CA_p, CB_p : array-like
        MO coefficients for the `p` index on shells A and B.
    CA_u, CB_u : array-like
        MO coefficients for the `u` index on shells A and B.
    same_shell : bool
        If True, assumes A == B.

    Returns
    -------
    array-like
        Matrix of shape `(nA*nB, np*nu)`.
    """

    xp = _get_xp(CA_p, CB_p, CA_u, CB_u)
    CA_p = xp.asarray(CA_p)
    CB_p = xp.asarray(CB_p)
    CA_u = xp.asarray(CA_u)
    CB_u = xp.asarray(CB_u)

    if CA_p.ndim != 2 or CB_p.ndim != 2 or CA_u.ndim != 2 or CB_u.ndim != 2:
        raise ValueError("CA_p/CB_p/CA_u/CB_u must be 2D arrays")
    if int(CA_p.shape[0]) != int(CA_u.shape[0]) or int(CB_p.shape[0]) != int(CB_u.shape[0]):
        raise ValueError("mixed pair coeffs require matching AO dimensions on each shell (CA_p vs CA_u, CB_p vs CB_u)")
    if int(CA_p.shape[1]) != int(CB_p.shape[1]):
        raise ValueError("CA_p and CB_p must have the same number of 'p' orbitals")
    if int(CA_u.shape[1]) != int(CB_u.shape[1]):
        raise ValueError("CA_u and CB_u must have the same number of 'u' orbitals")

    nA, n_p = map(int, CA_p.shape)
    nB = int(CB_p.shape[0])
    n_u = int(CA_u.shape[1])

    # K[μ,ν,p,u] = CA_p[μ,p] * CB_u[ν,u] (+ swapped term if A!=B)
    K = xp.einsum("ap,bu->abpu", CA_p, CB_u)
    if not bool(same_shell):
        K = K + xp.einsum("bp,au->abpu", CB_p, CA_u)
    return K.reshape((nA * nB, n_p * n_u))


def build_pair_coeff_packed(CA, CB, *, same_shell: bool):
    """Build packed pair coefficients, mapping AO products to packed MO pairs `pq` (p>=q).

    Similar to `build_pair_coeff_ordered`, but the MO dimension is reduced to
    `npair = norb*(norb+1)/2`.

    Parameters
    ----------
    CA, CB : array-like
        MO coefficients for shells A and B.
    same_shell : bool
        If True, A == B.

    Returns
    -------
    array-like
        Matrix of shape `(nA*nB, npair)`.
    """

    xp = _get_xp(CA, CB)
    CA = xp.asarray(CA)
    CB = xp.asarray(CB)
    if CA.ndim != 2 or CB.ndim != 2:
        raise ValueError("CA/CB must be 2D arrays with shape (nAO_in_shell, norb)")
    if CA.shape[1] != CB.shape[1]:
        raise ValueError("CA and CB must have the same norb (2nd dimension)")

    nA, norb = map(int, CA.shape)
    nB = int(CB.shape[0])
    tri_p, tri_q = xp.tril_indices(norb)
    K = xp.einsum("ap,bq->abpq", CA, CB)
    if not same_shell:
        K = K + xp.einsum("bp,aq->abpq", CB, CA)
    K = K[:, :, tri_p, tri_q]
    return K.reshape((nA * nB, int(tri_p.size)))


__all__ = [
    "build_pair_coeff_ordered",
    "build_pair_coeff_ordered_mixed",
    "build_pair_coeff_packed",
    "expand_eri_packed_to_ordered",
    "j_ps_from_eri_mat",
    "j_ps_from_eri_packed",
    "npair",
    "ordered_to_packed_index",
    "pair_id",
    "pair_id_matrix",
    "unpack_pair_id",
]
