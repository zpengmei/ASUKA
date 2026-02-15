from __future__ import annotations

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.oracle import _csr_for_epq, _get_epq_action_cache

_STEP_TO_OCC_F64 = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float64)  # E,U,L,D


def _molcas_p2lev_pairs(n: int) -> list[tuple[int, int]]:
    """OpenMolcas `MKTG3` pair-index ordering (P2LEV) for levels 1..n, converted to 0-based.

    `mktg3.f` enumerates ordered level pairs (IL,JL) in three contiguous blocks:
    1) IL < JL, 2) IL == JL, 3) IL > JL.
    """

    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    out: list[tuple[int, int]] = []
    for il in range(n - 1):
        for jl in range(il + 1, n):
            out.append((int(il), int(jl)))
    for il in range(n):
        out.append((int(il), int(il)))
    for il in range(1, n):
        for jl in range(il):
            out.append((int(il), int(jl)))
    if len(out) != n * n:
        raise RuntimeError("internal error: unexpected P2LEV length")
    return out

try:  # optional Cython in-place CSC @ dense kernels
    from asuka._epq_cy import (  # type: ignore[import-not-found]
        csc_matmul_dense_inplace_cy as _csc_matmul_dense_inplace_cy,
    )
except Exception:  # pragma: no cover
    _csc_matmul_dense_inplace_cy = None

try:  # optional SciPy-backed sparse matmul for E_pq applications
    from asuka.contract import _epq_spmat_list as _epq_spmat_list  # noqa: PLC0415
    from asuka.contract import _sp as _sp  # noqa: PLC0415
except Exception:  # pragma: no cover
    _sp = None
    _epq_spmat_list = None  # type: ignore[assignment]


def _fill_epq_vec(
    drt: DRT,
    cache,
    c: np.ndarray,
    *,
    p: int,
    q: int,
    out: np.ndarray,
) -> None:
    """Fill ``out[:] = (E_pq |c>)`` using the cached CSC-like representation."""

    p = int(p)
    q = int(q)
    if p == q:
        out[:] = _STEP_TO_OCC_F64[cache.steps[:, p]] * c
        return

    csr = _csr_for_epq(cache, drt, p, q)
    out.fill(0.0)
    indptr = csr.indptr
    indices = csr.indices
    data = csr.data
    for j in range(int(c.size)):
        cj = float(c[j])
        if cj == 0.0:
            continue
        start = int(indptr[j])
        end = int(indptr[j + 1])
        if start == end:
            continue
        out[indices[start:end]] += data[start:end] * cj


def _reorder_rdm_pyscf(dm1: np.ndarray, dm2: np.ndarray, *, inplace: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Match PySCF's `pyscf.fci.rdm.reorder_rdm` convention (2pdm delta correction + symmetrization).

    This is an internal helper and is intentionally treated as a private API.
    """

    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    n = int(dm1.shape[0])
    if dm1.shape != (n, n):
        raise ValueError("dm1 must be square")
    if dm2.shape != (n, n, n, n):
        raise ValueError("dm2 must have shape (n,n,n,n)")

    if not inplace:
        dm2 = dm2.copy()

    for k in range(n):
        dm2[:, k, k, :] -= dm1.T

    dm2 = 0.5 * (dm2 + dm2.transpose(2, 3, 0, 1))
    return dm1, np.asarray(dm2, dtype=np.float64, order="C")


def _reorder_dm123_pyscf(
    dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray, *, inplace: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match PySCF's `pyscf.fci.rdm.reorder_dm123` convention.

    This is an internal helper and is intentionally treated as a private API.
    """

    dm1, dm2 = _reorder_rdm_pyscf(dm1, dm2, inplace=inplace)
    if not inplace:
        dm3 = np.asarray(dm3, dtype=np.float64).copy()
    else:
        dm3 = np.asarray(dm3, dtype=np.float64)

    n = int(dm1.shape[0])
    if dm3.shape != (n, n, n, n, n, n):
        raise ValueError("dm3 must have shape (n,n,n,n,n,n)")

    for q in range(n):
        dm3[:, q, q, :, :, :] -= dm2
        dm3[:, :, :, q, q, :] -= dm2
        dm3[:, q, :, :, q, :] -= dm2.transpose(0, 2, 3, 1)
        for s in range(n):
            dm3[:, q, q, s, s, :] -= dm1.T
    return dm1, dm2, np.asarray(dm3, dtype=np.float64, order="C")


def _reorder_rdm_pyscf_trans(
    dm1: np.ndarray, dm2: np.ndarray, *, inplace: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Delta-correct a transition dm2 without enforcing the bra=ket symmetrization.

    This is the transition analogue of PySCF's `pyscf.fci.rdm.reorder_rdm`, but *without*
    applying the `0.5*(dm2 + dm2.transpose(2,3,0,1))` hermitization step (which is only
    valid for bra=ket densities).

    Notes
    -----
    The input tensors are assumed to match the raw dm2 conventions produced by
    `_make_rdm123_pyscf(..., reorder=False)` / PySCF's `make_dm123`.
    """

    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    n = int(dm1.shape[0])
    if dm1.shape != (n, n):
        raise ValueError("dm1 must be square")
    if dm2.shape != (n, n, n, n):
        raise ValueError("dm2 must have shape (n,n,n,n)")

    if not inplace:
        dm2 = dm2.copy()

    for k in range(n):
        dm2[:, k, k, :] -= dm1.T
    return dm1, np.asarray(dm2, dtype=np.float64, order="C")


def _reorder_dm123_pyscf_trans(
    dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray, *, inplace: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Delta-correct transition (dm2, dm3) without bra=ket symmetrization.

    This is the transition analogue of `_reorder_dm123_pyscf`.
    """

    dm1, dm2 = _reorder_rdm_pyscf_trans(dm1, dm2, inplace=inplace)
    if not inplace:
        dm3 = np.asarray(dm3, dtype=np.float64).copy()
    else:
        dm3 = np.asarray(dm3, dtype=np.float64)

    n = int(dm1.shape[0])
    if dm3.shape != (n, n, n, n, n, n):
        raise ValueError("dm3 must have shape (n,n,n,n,n,n)")

    for q in range(n):
        dm3[:, q, q, :, :, :] -= dm2
        dm3[:, :, :, q, q, :] -= dm2
        dm3[:, q, :, :, q, :] -= dm2.transpose(0, 2, 3, 1)
        for s in range(n):
            dm3[:, q, q, s, s, :] -= dm1.T
    return dm1, dm2, np.asarray(dm3, dtype=np.float64, order="C")


def _reorder_dm123_molcas_trans(
    dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray, *, inplace: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply OpenMolcas-style normal-order delta corrections to transition (dm2, dm3).

    This mirrors the correction logic in OpenMolcas `src/caspt2/mktg3.f` for
    transition densities, where:
      - dm2(T,U,V,X) := <E(TU)E(VX)> - δ(V,U) * dm1(T,X)
      - dm3(T,U,V,X,Y,Z) := <E(TU)E(VX)E(YZ)> with the corresponding 3-body delta terms

    Unlike `_reorder_dm123_pyscf_trans`, this uses `dm1` (not `dm1.T`) in the delta
    contractions and uses the (V,X,T,Z) ordering for the `δ(Y,U)` correction term.
    """

    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)

    n = int(dm1.shape[0])
    if dm1.shape != (n, n):
        raise ValueError("dm1 must be square")
    if dm2.shape != (n, n, n, n):
        raise ValueError("dm2 must have shape (n,n,n,n)")
    if dm3.shape != (n, n, n, n, n, n):
        raise ValueError("dm3 must have shape (n,n,n,n,n,n)")

    if not inplace:
        dm2 = dm2.copy()
        dm3 = dm3.copy()

    # 2-body delta correction (no bra=ket hermitization):
    #   <E_tu E_vx> -> <E_tu E_vx> - δ(v,u) <E_tx>
    for k in range(n):
        dm2[:, k, k, :] -= dm1

    # OpenMolcas `MKTG3` enforces pair-permutation symmetry of TG2 by *copying* the canonical
    # half (IP2 <= IP1 in its P2LEV pair-index ordering), not by averaging the two orderings.
    #
    # For transition densities the unsymmetrized TG2 depends on the chosen canonical half,
    # so we must mirror the P2LEV ordering rather than the naive lexicographic flattening.
    n2 = int(n * n)
    m = np.asarray(dm2, dtype=np.float64).reshape(n2, n2, order="F")  # pair exchange is matrix transpose

    perm = np.asarray([il + n * jl for il, jl in _molcas_p2lev_pairs(n)], dtype=np.int64)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(n2, dtype=np.int64)

    m_p = m[np.ix_(perm, perm)]
    m_p_lo = np.tril(m_p)
    m_p_sym = m_p_lo + np.tril(m_p, -1).T
    m_sym = m_p_sym[np.ix_(inv, inv)]
    dm2 = np.asarray(m_sym.reshape(n, n, n, n, order="F"), dtype=np.float64, order="C")

    # 3-body delta correction (no bra=ket symmetrization):
    #   <E_tu E_vx E_yz> minus the contractions described in `mktg3.f`.
    dm2_vxtz = dm2.transpose(2, 0, 1, 3)  # (t,v,x,z) = dm2[v,x,t,z]
    for q in range(n):
        dm3[:, q, q, :, :, :] -= dm2
        dm3[:, :, :, q, q, :] -= dm2
        dm3[:, q, :, :, q, :] -= dm2_vxtz
        for s in range(n):
            dm3[:, q, q, s, s, :] -= dm1
    # OpenMolcas stores TG3 with full permutation symmetry of the three (tu),(vx),(yz) pairs.
    dm3_sym = np.asarray(dm3, dtype=np.float64, order="C").copy()
    dm3_sym += dm3.transpose(2, 3, 0, 1, 4, 5)
    dm3_sym += dm3.transpose(4, 5, 2, 3, 0, 1)
    dm3_sym += dm3.transpose(0, 1, 4, 5, 2, 3)
    dm3_sym += dm3.transpose(2, 3, 4, 5, 0, 1)
    dm3_sym += dm3.transpose(4, 5, 0, 1, 2, 3)
    dm3 = np.asarray(dm3_sym * (1.0 / 6.0), dtype=np.float64, order="C")

    return dm1, dm2, dm3


def _reorder_dm123_molcas(
    dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray, *, inplace: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply OpenMolcas-style normal-order delta corrections to diagonal (bra=ket) (dm2, dm3).

    OpenMolcas constructs the SavGradParams `GAMMA2/GAMMA3` (and `DELTA2/DELTA3`)
    stored in `PT2GRD` for *state* densities via `MKFG3`:
      - `OpenMolcas/src/caspt2/mkfg3.f`

    The normal-order delta corrections match `MKTG3`:
      - `OpenMolcas/src/caspt2/mktg3.f`

    For diagonal (real) densities, `MKFG3` additionally enforces the Hermitian
    reversal symmetry of the 2-body tensor:

      GAMMA2(t,u,v,x) = GAMMA2(x,v,u,t)

    and the pair-exchange symmetry:

      GAMMA2(t,u,v,x) = GAMMA2(v,x,t,u)

    This routine mirrors that convention by:
      1) applying the Molcas delta corrections to raw <E_tu E_vx> / <E_tu E_vx E_yz>,
      2) enforcing the MKFG3 diagonal symmetries for `GAMMA2`,
      3) using that `GAMMA2` in the 3-body delta-correction loop,
      4) enforcing full permutation symmetry of the three (tu),(vx),(yz) pairs for `GAMMA3`.
    """

    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)

    n = int(dm1.shape[0])
    if dm1.shape != (n, n):
        raise ValueError("dm1 must be square")
    if dm2.shape != (n, n, n, n):
        raise ValueError("dm2 must have shape (n,n,n,n)")
    if dm3.shape != (n, n, n, n, n, n):
        raise ValueError("dm3 must have shape (n,n,n,n,n,n)")

    if not inplace:
        dm2 = dm2.copy()
        dm3 = dm3.copy()

    # 2-body delta correction:
    #   <E_tu E_vx> -> <E_tu E_vx> - δ(v,u) <E_tx>
    for k in range(n):
        dm2[:, k, k, :] -= dm1

    # MKFG3 diagonal symmetries (real bra=ket):
    #   (tu,vx) == (vx,tu)  and  (tu,vx) == (xv,ut)
    dm2 = np.asarray(
        0.25
        * (
            dm2
            + dm2.transpose(2, 3, 0, 1)  # pair exchange
            + dm2.transpose(3, 2, 1, 0)  # Hermitian reversal
            + dm2.transpose(1, 0, 3, 2)  # implied: swap within pairs
        ),
        dtype=np.float64,
        order="C",
    )

    # 3-body delta correction:
    #   <E_tu E_vx E_yz> minus the contractions described in `mktg3.f` / `mkfg3.f`.
    dm2_vxtz = dm2.transpose(2, 0, 1, 3)  # (t,v,x,z) = dm2[v,x,t,z]
    for q in range(n):
        # v == u
        dm3[:, q, q, :, :, :] -= dm2
        # y == x
        dm3[:, :, :, q, q, :] -= dm2
        # y == u
        dm3[:, q, :, :, q, :] -= dm2_vxtz
        # (v == u) and (y == x)
        for s in range(n):
            dm3[:, q, q, s, s, :] -= dm1

    # OpenMolcas stores TG3/G3 with full permutation symmetry of the three (tu),(vx),(yz) pairs.
    dm3_sym = np.asarray(dm3, dtype=np.float64, order="C").copy()
    dm3_sym += dm3.transpose(2, 3, 0, 1, 4, 5)
    dm3_sym += dm3.transpose(4, 5, 2, 3, 0, 1)
    dm3_sym += dm3.transpose(0, 1, 4, 5, 2, 3)
    dm3_sym += dm3.transpose(2, 3, 4, 5, 0, 1)
    dm3_sym += dm3.transpose(4, 5, 0, 1, 2, 3)
    dm3 = np.asarray(dm3_sym * (1.0 / 6.0), dtype=np.float64, order="C")

    return dm1, dm2, dm3


def _make_rdm123_pyscf(
    drt: DRT,
    civec: np.ndarray,
    *,
    max_memory_mb: float = 4000.0,
    reorder: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (dm1, dm2, dm3) for a CSF wavefunction in PySCF's `make_rdm123` conventions.

    This returns the *reordered* (default) spin-traced RDMs matching:
      `pyscf.fci.direct_spin1.make_rdm123(reorder=True)`.

    Notes
    -----
    The raw tensors correspond to alternating creation/annihilation indices:
    - dm1[p,q] = <p^+ q>
    - dm2[p,q,r,s] = <p^+ q r^+ s>
    - dm3[p,q,r,s,t,u] = <p^+ q r^+ s t^+ u>

    Setting `reorder=True` applies PySCF's `reorder_dm123` transformation, which
    inserts the delta-term contractions to yield the normal-ordered density
    matrices used by PySCF MRPT routines.
    """

    ncsf = int(drt.ncsf)
    norb = int(drt.norb)
    nops = norb * norb

    c = np.asarray(civec, dtype=np.float64).ravel()
    if c.size != ncsf:
        raise ValueError("civec has wrong length")

    max_memory_bytes = float(max_memory_mb) * 1e6
    if max_memory_bytes <= 0:
        max_memory_bytes = 1.0

    # Memory guard: dm3 is the dominant object.
    dm3_bytes = float(nops) * float(nops) * float(nops) * 8.0
    if dm3_bytes > max_memory_bytes:
        raise MemoryError(
            f"dm3 allocation would require ~{dm3_bytes/1e6:.1f} MB "
            f"(norb={norb}, ncsf={ncsf}); increase max_memory_mb or use a smaller active space"
        )

    cache = _get_epq_action_cache(drt)
    occ = np.asarray(_STEP_TO_OCC_F64[cache.steps], dtype=np.float64, order="C")

    mats = None
    if _sp is not None and _epq_spmat_list is not None:
        mats = _epq_spmat_list(drt, cache)

    # Build T[pq, :] = E_pq |c> (row-major in pq).
    T = np.empty((nops, ncsf), dtype=np.float64)
    c_col = c.reshape(ncsf, 1)
    for p in range(norb):
        for q in range(norb):
            pq = p * norb + q
            out = T[pq]
            if p == q:
                np.multiply(occ[:, p], c, out=out)
            elif mats is not None:
                mat = mats[pq]
                if mat is None:
                    raise AssertionError("missing E_pq sparse matrix")
                if _csc_matmul_dense_inplace_cy is not None:
                    _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                        mat.indptr, mat.indices, mat.data, c_col, out.reshape(ncsf, 1)
                    )
                else:
                    out[:] = mat.dot(c)  # type: ignore[operator]
            else:
                _fill_epq_vec(drt, cache, c, p=p, q=q, out=out)

    # Raw dm1[p,q] = <E_pq>.
    dm1 = (T @ c).reshape(norb, norb)

    # Dense matrix of ket vectors: X[:, tu] = E_tu |c>.
    X = np.ascontiguousarray(T.T)  # (ncsf, nops)

    # Bra block B[pq, :] = E_qp |c> (view; no copy).
    B = T.reshape(norb, norb, ncsf).transpose(1, 0, 2).reshape(nops, ncsf)

    # Raw dm2 and dm3.
    dm2_flat = B @ X  # (nops, nops) = <E_pq E_rs> in flattened pair space
    dm2 = dm2_flat.reshape(norb, norb, norb, norb)

    dm3_flat = np.empty((nops, nops, nops), dtype=np.float64)
    Y = np.empty((ncsf, nops), dtype=np.float64)
    M = np.empty((nops, nops), dtype=np.float64)
    for r in range(norb):
        for s in range(norb):
            rs = r * norb + s
            if r == s:
                np.multiply(occ[:, r][:, None], X, out=Y)
            elif mats is not None:
                mat = mats[rs]
                if mat is None:
                    raise AssertionError("missing E_pq sparse matrix")
                if _csc_matmul_dense_inplace_cy is not None:
                    _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                        mat.indptr, mat.indices, mat.data, X, Y
                    )
                else:
                    Y[:] = mat.dot(X)  # type: ignore[operator]
            else:
                for tu in range(nops):
                    tvec = X[:, tu]
                    out = Y[:, tu]
                    _fill_epq_vec(drt, cache, tvec, p=r, q=s, out=out)

            np.matmul(B, Y, out=M)
            dm3_flat[:, rs, :] = M

    dm3 = dm3_flat.reshape(norb, norb, norb, norb, norb, norb)

    if reorder:
        dm1, dm2, dm3 = _reorder_dm123_pyscf(dm1, dm2, dm3, inplace=True)
    else:
        dm1 = np.asarray(dm1, dtype=np.float64, order="C")
        dm2 = np.asarray(dm2, dtype=np.float64, order="C")
        dm3 = np.asarray(dm3, dtype=np.float64, order="C")

    return dm1, dm2, dm3


def _trans_rdm123_pyscf(
    drt: DRT,
    ci_bra: np.ndarray,
    ci_ket: np.ndarray,
    *,
    max_memory_mb: float = 4000.0,
    reorder: bool = False,
    reorder_mode: str = "pyscf",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute transition (dm1, dm2, dm3) between two CSF wavefunctions in PySCF-like conventions.

    Parameters
    ----------
    drt:
        Active-space DRT describing the CSF basis.
    ci_bra, ci_ket:
        CSF coefficient vectors (length `drt.ncsf`).
    max_memory_mb:
        Guardrail for the dominant `dm3` allocation.
    reorder:
        If True, apply the same delta-term contractions as PySCF's `reorder_dm123`, but
        **without** the bra=ket symmetrization of dm2.
    reorder_mode:
        When `reorder=True`, choose the delta-correction convention:
          - "pyscf": `_reorder_dm123_pyscf_trans` (default; matches PySCF's index order)
          - "molcas": `_reorder_dm123_molcas_trans` (matches OpenMolcas `mktg3.f` for transitions)

    Returns
    -------
    (dm1, dm2, dm3):
        Raw spin-traced tensors matching `_make_rdm123_pyscf(..., reorder=False)` when
        `ci_bra == ci_ket`.

    Notes
    -----
    This routine is primarily intended for contracted quantities (e.g. OpenMolcas-style
    DELTA/B-matrix machinery) where the bra and ket differ (e.g. `|ket> = H0diag|Ψ>`).
    """

    ncsf = int(drt.ncsf)
    norb = int(drt.norb)
    nops = norb * norb

    cbra = np.asarray(ci_bra, dtype=np.float64).ravel()
    cket = np.asarray(ci_ket, dtype=np.float64).ravel()
    if cbra.size != ncsf or cket.size != ncsf:
        raise ValueError("ci_bra/ci_ket have wrong length")

    max_memory_bytes = float(max_memory_mb) * 1e6
    if max_memory_bytes <= 0:
        max_memory_bytes = 1.0

    dm3_bytes = float(nops) * float(nops) * float(nops) * 8.0
    if dm3_bytes > max_memory_bytes:
        raise MemoryError(
            f"dm3 allocation would require ~{dm3_bytes/1e6:.1f} MB "
            f"(norb={norb}, ncsf={ncsf}); increase max_memory_mb or use a smaller active space"
        )

    cache = _get_epq_action_cache(drt)
    occ = np.asarray(_STEP_TO_OCC_F64[cache.steps], dtype=np.float64, order="C")

    mats = None
    if _sp is not None and _epq_spmat_list is not None:
        mats = _epq_spmat_list(drt, cache)

    # Ket block: T_ket[pq, :] = E_pq |ket>.
    T_ket = np.empty((nops, ncsf), dtype=np.float64)
    cket_col = cket.reshape(ncsf, 1)
    for p in range(norb):
        for q in range(norb):
            pq = p * norb + q
            out = T_ket[pq]
            if p == q:
                np.multiply(occ[:, p], cket, out=out)
            elif mats is not None:
                mat = mats[pq]
                if mat is None:
                    raise AssertionError("missing E_pq sparse matrix")
                if _csc_matmul_dense_inplace_cy is not None:
                    _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                        mat.indptr, mat.indices, mat.data, cket_col, out.reshape(ncsf, 1)
                    )
                else:
                    out[:] = mat.dot(cket)  # type: ignore[operator]
            else:
                _fill_epq_vec(drt, cache, cket, p=p, q=q, out=out)

    # Raw dm1[p,q] = <bra|E_pq|ket>.
    dm1 = (T_ket @ cbra).reshape(norb, norb)

    # Bra block: B_bra[pq, :] = E_qp |bra> (row labeled by pq).
    B_bra = np.empty((nops, ncsf), dtype=np.float64)
    cbra_col = cbra.reshape(ncsf, 1)
    for p in range(norb):
        for q in range(norb):
            pq = p * norb + q
            out = B_bra[pq]
            if p == q:
                np.multiply(occ[:, p], cbra, out=out)
            elif mats is not None:
                mat = mats[q * norb + p]
                if mat is None:
                    raise AssertionError("missing E_pq sparse matrix")
                if _csc_matmul_dense_inplace_cy is not None:
                    _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                        mat.indptr, mat.indices, mat.data, cbra_col, out.reshape(ncsf, 1)
                    )
                else:
                    out[:] = mat.dot(cbra)  # type: ignore[operator]
            else:
                _fill_epq_vec(drt, cache, cbra, p=q, q=p, out=out)

    X = np.ascontiguousarray(T_ket.T)  # (ncsf, nops)
    dm2_flat = B_bra @ X
    dm2 = dm2_flat.reshape(norb, norb, norb, norb)

    dm3_flat = np.empty((nops, nops, nops), dtype=np.float64)
    Y = np.empty((ncsf, nops), dtype=np.float64)
    M = np.empty((nops, nops), dtype=np.float64)
    for r in range(norb):
        for s in range(norb):
            rs = r * norb + s
            if r == s:
                np.multiply(occ[:, r][:, None], X, out=Y)
            elif mats is not None:
                mat = mats[rs]
                if mat is None:
                    raise AssertionError("missing E_pq sparse matrix")
                if _csc_matmul_dense_inplace_cy is not None:
                    _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                        mat.indptr, mat.indices, mat.data, X, Y
                    )
                else:
                    Y[:] = mat.dot(X)  # type: ignore[operator]
            else:
                for tu in range(nops):
                    tvec = X[:, tu]
                    out = Y[:, tu]
                    _fill_epq_vec(drt, cache, tvec, p=r, q=s, out=out)

            np.matmul(B_bra, Y, out=M)
            dm3_flat[:, rs, :] = M

    dm3 = dm3_flat.reshape(norb, norb, norb, norb, norb, norb)

    if bool(reorder):
        mode = str(reorder_mode).strip().lower()
        if mode == "pyscf":
            dm1, dm2, dm3 = _reorder_dm123_pyscf_trans(dm1, dm2, dm3, inplace=True)
        elif mode == "molcas":
            dm1, dm2, dm3 = _reorder_dm123_molcas_trans(dm1, dm2, dm3, inplace=True)
        else:
            raise ValueError("reorder_mode must be 'pyscf' or 'molcas'")
    else:
        dm1 = np.asarray(dm1, dtype=np.float64, order="C")
        dm2 = np.asarray(dm2, dtype=np.float64, order="C")
        dm3 = np.asarray(dm3, dtype=np.float64, order="C")
    return dm1, dm2, dm3
