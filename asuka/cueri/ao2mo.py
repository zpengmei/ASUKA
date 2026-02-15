"""Native cuERI-based MO integral transforms replacing ``pyscf.ao2mo``.

Public API
----------
- ``ao2mo_kernel``  — full 4-index MO transform (single-set or 4-set)
- ``ao2mo_restore`` — unpack lower-triangular pair matrix to 4D tensor
- ``nr_e2_half_transform`` — half-transform AO-pair buffer to MO basis
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .active_space_dense_cpu import CuERIActiveSpaceDenseCPUBuilder
from .basis_cart import BasisCartSoA
from .mol_basis import get_cached_or_pack_cart_ao_basis, pack_cart_shells_from_mol


def _resolve_ao_basis(
    ao_basis_or_mol: BasisCartSoA | Any,
) -> BasisCartSoA:
    """Accept a ``BasisCartSoA`` directly or build one from a PySCF mol."""
    if isinstance(ao_basis_or_mol, BasisCartSoA):
        return ao_basis_or_mol
    return pack_cart_shells_from_mol(ao_basis_or_mol)


def _is_4tuple(mo_coeffs) -> bool:
    """Return True if *mo_coeffs* is a 4-tuple/list of arrays."""
    if isinstance(mo_coeffs, (tuple, list)) and len(mo_coeffs) == 4:
        return all(hasattr(c, "ndim") or isinstance(c, np.ndarray) for c in mo_coeffs)
    return False


# ---------------------------------------------------------------------------
# ao2mo_kernel — full 4-index MO transform
# ---------------------------------------------------------------------------

def ao2mo_kernel(
    ao_basis_or_mol: BasisCartSoA | Any,
    mo_coeffs: np.ndarray | Sequence[np.ndarray],
    *,
    compact: bool = False,
    builder: CuERIActiveSpaceDenseCPUBuilder | None = None,
) -> np.ndarray:
    """Perform a full 4-index molecular orbital (MO) integral transformation (pq|rs) via cuERI.

    This function transforms AO integrals to the MO basis using either a single set of
    coefficients or four distinct sets.

    Parameters
    ----------
    ao_basis_or_mol : BasisCartSoA | object
        The basis set definition. Can be a `BasisCartSoA` object or a PySCF `mol` object.
    mo_coeffs : np.ndarray | Sequence[np.ndarray]
        MO coefficients.
        - For a single set: A 2-D array of shape `(nao, nmo)`.
        - For a generalized transform: A sequence of 4 arrays `(C1, C2, C3, C4)`,
          each of shape `(nao, nmo_i)`.
    compact : bool, default=False
        If True and `mo_coeffs` is a single set, the output is returned in a packed
        lower-triangular format corresponding to `(npair, npair)` where
        `npair = nmo * (nmo + 1) / 2`. Ignored for 4-set transforms.
    builder : CuERIActiveSpaceDenseCPUBuilder | None, optional
        A pre-configured builder instance to reuse cached basis preprocessing artifacts.

    Returns
    -------
    np.ndarray
        The transformed integrals.
        - Single set, not compact: `(nmo^2, nmo^2)` matrix (ordered-pair layout).
        - Single set, compact: `(npair, npair)` matrix.
        - 4-set transform: `(n1*n2, n3*n4)` matrix.
    """
    ao_basis = _resolve_ao_basis(ao_basis_or_mol)
    if builder is None:
        builder = CuERIActiveSpaceDenseCPUBuilder(ao_basis=ao_basis)

    if _is_4tuple(mo_coeffs):
        return _ao2mo_general(builder, mo_coeffs)
    else:
        C = np.asarray(mo_coeffs, dtype=np.float64, order="C")
        if C.ndim != 2:
            raise ValueError("Single-set mo_coeffs must be a 2-D array")
        if compact:
            return builder.build_eri_packed(
                C, eps_ao=0.0, eps_mo=0.0, blas_nthreads=None, profile=None,
            )
        return builder.build_eri_mat(
            C, eps_ao=0.0, eps_mo=0.0, blas_nthreads=None, profile=None,
        )


def _ao2mo_general(
    builder: CuERIActiveSpaceDenseCPUBuilder,
    mo_coeffs: Sequence[np.ndarray],
) -> np.ndarray:
    """4-set transform via build-combined-then-slice strategy."""
    C1, C2, C3, C4 = (np.asarray(c, dtype=np.float64, order="C") for c in mo_coeffs)
    for i, c in enumerate((C1, C2, C3, C4)):
        if c.ndim != 2:
            raise ValueError(f"mo_coeffs[{i}] must be a 2-D array")

    nao = C1.shape[0]
    n1, n2, n3, n4 = C1.shape[1], C2.shape[1], C3.shape[1], C4.shape[1]

    # Build combined MO coefficient matrix with unique columns.
    C_all = np.concatenate([C1, C2, C3, C4], axis=1)  # (nao, n1+n2+n3+n4)
    nmo_all = C_all.shape[1]

    # Full ordered-pair ERI in the combined basis: (nmo_all^2, nmo_all^2)
    eri_full = builder.build_eri_mat(
        C_all, eps_ao=0.0, eps_mo=0.0, blas_nthreads=None, profile=None,
    )

    # Index ranges in the combined basis.
    off = 0
    idx1 = np.arange(off, off + n1); off += n1
    idx2 = np.arange(off, off + n2); off += n2
    idx3 = np.arange(off, off + n3); off += n3
    idx4 = np.arange(off, off + n4); off += n4

    # Ordered-pair indices: row = (p in C1) * nmo_all + (q in C2)
    row_ids = (idx1[:, None] * nmo_all + idx2[None, :]).ravel()
    col_ids = (idx3[:, None] * nmo_all + idx4[None, :]).ravel()

    return np.ascontiguousarray(eri_full[np.ix_(row_ids, col_ids)])


# ---------------------------------------------------------------------------
# ao2mo_restore — unpack triangular pair matrix to 4D
# ---------------------------------------------------------------------------

def ao2mo_restore(symmetry: int, eri_packed: np.ndarray, norb: int) -> np.ndarray:
    """Unpack a lower-triangular packed ERI matrix to a full 4-D tensor.

    Restores the full 4-D tensor `(norb, norb, norb, norb)` from a packed representation.

    Parameters
    ----------
    symmetry : int
        Symmetry flag. Currently, only `symmetry=1` (no symmetry/full restore) is supported.
    eri_packed : np.ndarray
        The packed ERI matrix of shape `(npair, npair)` where `npair = norb * (norb + 1) / 2`.
    norb : int
        The number of orbitals.

    Returns
    -------
    np.ndarray
        The full 4-D ERI tensor of shape `(norb, norb, norb, norb)`.

    Raises
    ------
    NotImplementedError
        If `symmetry` is not 1.
    """
    if int(symmetry) != 1:
        raise NotImplementedError("Only symmetry=1 (full 4D restore) is supported")

    from .eri_utils import expand_eri_packed_to_ordered  # noqa: PLC0415

    eri_packed = np.asarray(eri_packed, dtype=np.float64)
    eri_mat = expand_eri_packed_to_ordered(eri_packed, norb)
    return eri_mat.reshape(norb, norb, norb, norb)


# ---------------------------------------------------------------------------
# nr_e2_half_transform — AO-pair → MO half-transform
# ---------------------------------------------------------------------------

def nr_e2_half_transform(
    eri_ao: np.ndarray,
    mo_coeff: np.ndarray,
    mo_slice: tuple[int, int] | tuple[int, int, int, int],
    *,
    aosym: str = "s1",
    mosym: str = "s1",
) -> np.ndarray:
    """Perform a half-transformation of AO-pair integrals to the MO basis.

    This function contracts the last two AO indices of the input integrals with MO coefficients,
    effectively transforming (kl|uv) -> (kl|pq) or similar partial transforms.

    Parameters
    ----------
    eri_ao : np.ndarray
        The input AO integrals. Shape: `(nrow, nao_pair)`.
        - `nao_pair = nao * nao` if `aosym='s1'`.
        - `nao_pair = nao * (nao + 1) / 2` if `aosym='s2kl'` or `'s2'`.
    mo_coeff : np.ndarray
        The MO coefficients. Shape: `(nao, nmo_total)`.
    mo_slice : tuple[int, int] | tuple[int, int, int, int]
        Specifies the subset of MOs to transform.
        - 2-tuple `(i, j)`: Transforms to `mo_coeff[:, i:j]` for both indices.
        - 4-tuple `(i, j, k, l)`: Transforms the first index to `mo_coeff[:, i:j]` and
          the second to `mo_coeff[:, k:l]`.
    aosym : str, default='s1'
        Symmetry of the input AO pairs.
        - 's1': Square (no symmetry), shape `(nrow, nao^2)`.
        - 's2kl', 's2': Lower-triangular packed, shape `(nrow, nao*(nao+1)/2)`.
    mosym : str, default='s1'
        Symmetry of the output MO pairs.
        - 's1': Square (no symmetry), shape `(nrow, nmo_i * nmo_j)`.
        - 's2': Lower-triangular packed, shape `(nrow, nmo*(nmo+1)/2)`. Only valid if
          the two MO ranges are identical.

    Returns
    -------
    np.ndarray
        The half-transformed integrals.
        Shape depends on `mosym`:
        - 's1': `(nrow, nmo_i * nmo_j)`
        - 's2': `(nrow, nmo * (nmo + 1) / 2)`
    """
    eri_ao = np.asarray(eri_ao, dtype=np.float64)
    mo_coeff = np.asarray(mo_coeff, dtype=np.float64)

    if len(mo_slice) == 4:
        i, j, k, l = (int(x) for x in mo_slice)
        Ci = np.ascontiguousarray(mo_coeff[:, i:j])
        Cj = np.ascontiguousarray(mo_coeff[:, k:l])
    elif len(mo_slice) == 2:
        i, j = int(mo_slice[0]), int(mo_slice[1])
        Ci = Cj = np.ascontiguousarray(mo_coeff[:, i:j])
    else:
        raise ValueError("mo_slice must be a 2-tuple or 4-tuple")

    nao = Ci.shape[0]
    nmo_i = Ci.shape[1]
    nmo_j = Cj.shape[1]
    nrow = eri_ao.shape[0]

    # Unpack AO pairs to square if needed.
    if aosym in ("s2kl", "s2"):
        nao_pair = nao * (nao + 1) // 2
        if eri_ao.shape[1] != nao_pair:
            raise ValueError(
                f"aosym='{aosym}' expects nao_pair={nao_pair}, got {eri_ao.shape[1]}"
            )
        eri_sq = _unpack_tril_rows(eri_ao, nao)
    elif aosym == "s1":
        if eri_ao.shape[1] != nao * nao:
            raise ValueError(
                f"aosym='s1' expects nao*nao={nao*nao}, got {eri_ao.shape[1]}"
            )
        eri_sq = eri_ao.reshape(nrow, nao, nao)
    else:
        raise ValueError(f"Unsupported aosym='{aosym}'")

    # Contract: out[row, p, q] = sum_{mu,nu} eri_sq[row, mu, nu] * Ci[mu,p] * Cj[nu,q]
    tmp = np.einsum("rmn,nq->rmq", eri_sq, Cj)
    out = np.einsum("rmq,mp->rpq", tmp, Ci)

    if mosym in ("s2",):
        nmo = nmo_i  # s2 packing only valid when nmo_i == nmo_j
        tri_p, tri_q = np.tril_indices(nmo)
        return np.ascontiguousarray(out[:, tri_p, tri_q])
    return out.reshape(nrow, nmo_i * nmo_j)


def _unpack_tril_rows(eri_tril: np.ndarray, nao: int) -> np.ndarray:
    """Unpack lower-triangular AO pairs to square for each row.

    Parameters
    ----------
    eri_tril : np.ndarray
        Shape ``(nrow, nao*(nao+1)/2)``.
    nao : int
        Number of AOs.

    Returns
    -------
    np.ndarray
        Shape ``(nrow, nao, nao)``.
    """
    nrow = eri_tril.shape[0]
    out = np.empty((nrow, nao, nao), dtype=eri_tril.dtype)
    tri_r, tri_c = np.tril_indices(nao)
    out[:, tri_r, tri_c] = eri_tril
    out[:, tri_c, tri_r] = eri_tril
    return out


__all__ = [
    "ao2mo_kernel",
    "ao2mo_restore",
    "nr_e2_half_transform",
]
