from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def pair_index_s2(p: int, q: int) -> int:
    """Packed lower-triangular (s2) pair index for an unordered pair (p,q).

    This matches the common PySCF convention for "s2" packed pair ordering:
      idx(p,q) = p*(p+1)//2 + q  (with p >= q).
    """

    p = int(p)
    q = int(q)
    if p < q:
        p, q = q, p
    return p * (p + 1) // 2 + q


def _pair_index_full(norb: int, p: int, q: int) -> int:
    return int(p) * int(norb) + int(q)


def _npair_s2(norb: int) -> int:
    norb = int(norb)
    return norb * (norb + 1) // 2


def _as_f64_2d(a: Any) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("expected a 2D array")
    return arr


def _expand_cderi_s2_to_full_pairs(cderi_s2: np.ndarray, norb: int) -> np.ndarray:
    """Expand packed s2 CDERIs to full ordered-pair matrix.

    Parameters
    ----------
    cderi_s2:
        MO-basis CDERIs in packed s2 pair format. Accepted shapes:
          - (naux, npair_s2)
          - (npair_s2, naux)

    Returns
    -------
    l_full:
        Array of shape (norb*norb, naux) where row (p*norb+q) stores the
        length-naux Cholesky vector d[L,pq] for the (ordered) MO pair (p,q).
        Rows for (p,q) and (q,p) are identical.
    """

    norb = int(norb)
    cderi_s2 = _as_f64_2d(cderi_s2)
    expected_pairs = _npair_s2(norb)

    if cderi_s2.shape[1] == expected_pairs:
        pass
    elif cderi_s2.shape[0] == expected_pairs:
        cderi_s2 = cderi_s2.T
    else:
        raise ValueError(f"unexpected cderi_s2 shape {cderi_s2.shape} for norb={norb}")

    # Vectorized expansion: map ordered (p,q) pairs to packed s2 pair indices.
    naux = int(cderi_s2.shape[0])
    pair_id = np.empty((norb, norb), dtype=np.intp)
    for p in range(norb):
        for q in range(norb):
            pair_id[p, q] = pair_index_s2(p, q)
    flat = pair_id.reshape(norb * norb)
    return np.asarray(cderi_s2[:, flat].T, dtype=np.float64, order="C").reshape(norb * norb, naux)


@dataclass(frozen=True)
class DFMOIntegrals:
    """DF/Cholesky-vector representation of (pq|rs) in an MO subspace.

    Stores vectors d[L,pq] such that:
      (pq|rs) ~= sum_L d[L,pq] d[L,rs] = dot(d[:,pq], d[:,rs])

    For convenience/performance, `l_full` stores the (ordered) MO-pair vectors in
    shape (norb*norb, naux).
    """

    norb: int
    l_full: np.ndarray  # (norb*norb, naux)
    j_ps: np.ndarray  # (norb, norb), J_{ps} = sum_q (p q| q s)
    pair_norm: np.ndarray  # (norb*norb,), ||d[:,pq]||_2 for ordered MO pairs
    _eri_mat: np.ndarray | None = None  # optional cache: (norb*norb, norb*norb)

    @property
    def naux(self) -> int:
        return int(self.l_full.shape[1])

    def _maybe_build_eri_mat(self, eri_mat_max_bytes: int) -> np.ndarray | None:
        """Build and cache the DF ERI matrix in pair space, if allowed by `eri_mat_max_bytes`."""

        eri_mat_max_bytes = int(eri_mat_max_bytes)
        if eri_mat_max_bytes <= 0:
            return None
        if self._eri_mat is not None:
            return self._eri_mat

        norb = int(self.norb)
        nops = norb * norb
        need = nops * nops * np.dtype(np.float64).itemsize
        if need > eri_mat_max_bytes:
            return None

        eri_mat = self.l_full @ self.l_full.T
        eri_mat_c = np.asarray(eri_mat, order="C")
        object.__setattr__(self, "_eri_mat", eri_mat_c)
        return eri_mat_c

    @classmethod
    def from_mo_cderi_s2(cls, mo_cderi_s2: np.ndarray, norb: int) -> DFMOIntegrals:
        norb = int(norb)
        l_full = None
        pair_norm = None
        try:
            from asuka._epq_cy import (  # type: ignore[import-not-found]
                expand_cderi_s2_to_full_pairs_and_norm_cy as _expand_cderi_s2_to_full_pairs_and_norm_cy,
            )
        except Exception:  # pragma: no cover
            _expand_cderi_s2_to_full_pairs_and_norm_cy = None

        if _expand_cderi_s2_to_full_pairs_and_norm_cy is not None:
            l_full, pair_norm = _expand_cderi_s2_to_full_pairs_and_norm_cy(mo_cderi_s2, norb)
        else:
            l_full = _expand_cderi_s2_to_full_pairs(mo_cderi_s2, norb=norb)
            l_full = np.asarray(l_full, dtype=np.float64, order="C")
            pair_norm = np.linalg.norm(l_full, axis=1)

        l3 = l_full.reshape(norb, norb, -1)
        # J_{ps} = sum_q (p q| q s) ~= sum_{q,L} d[L,pq] d[L,qs]
        j_ps = np.einsum("pql,qsl->ps", l3, l3, optimize=True)
        return cls(
            norb=norb,
            l_full=np.asarray(l_full, order="C"),
            j_ps=np.asarray(j_ps, order="C"),
            pair_norm=np.asarray(pair_norm, dtype=np.float64, order="C"),
        )

    def eri_pqrs(self, p: int, q: int, r: int, s: int) -> float:
        """Return (pq|rs) in chemist notation for MO indices in [0,norb)."""

        norb = int(self.norb)
        pq = _pair_index_full(norb, int(p), int(q))
        rs = _pair_index_full(norb, int(r), int(s))
        return float(np.dot(self.l_full[pq], self.l_full[rs]))

    def v_ijkl(self, i: int, j: int, k: int, l: int) -> float:
        """Return Dobrautz V_{ij,kl} = (i k| j l) with chemist ERI convention."""

        return self.eri_pqrs(int(i), int(k), int(j), int(l))

    def to_eri4(self) -> np.ndarray:
        """Materialize the approximate 4-index tensor eri[p,q,r,s]=(pq|rs).

        Intended for validation on tiny systems only.
        """

        norb = int(self.norb)
        eri = np.empty((norb, norb, norb, norb), dtype=np.float64)
        for p in range(norb):
            for q in range(norb):
                pq = _pair_index_full(norb, p, q)
                v_pq = self.l_full[pq]
                for r in range(norb):
                    for s in range(norb):
                        rs = _pair_index_full(norb, r, s)
                        eri[p, q, r, s] = float(np.dot(v_pq, self.l_full[rs]))
        return eri

    def weighted_pair_sum(self, pair_ids: np.ndarray, coeff: np.ndarray) -> np.ndarray:
        """Return w[L] = sum_t coeff[t] * d[L, pair_ids[t]]."""

        pair_ids = np.asarray(pair_ids, dtype=np.int32).ravel()
        coeff = np.asarray(coeff, dtype=np.float64).ravel()
        if pair_ids.size != coeff.size:
            raise ValueError("pair_ids and coeff must have same length")
        if pair_ids.size == 0:
            return np.zeros(int(self.naux), dtype=np.float64)
        # Avoid an explicit transpose (which can trigger extra copies for small-ish arrays):
        #   (L[pairs].T @ coeff) == (coeff @ L[pairs])
        return coeff @ self.l_full[pair_ids]

    def contract_cols(
        self,
        pair_ids: np.ndarray,
        coeff: np.ndarray,
        *,
        half: float = 0.5,
        eri_mat_max_bytes: int = 0,
    ) -> np.ndarray:
        """Return g[pq] = half * sum_{rs in pair_ids} (pq|rs) coeff[rs].

        Notes
        -----
        If `eri_mat_max_bytes > 0` and the implied matrix size fits, this function may
        cache and use an explicit DF ERI matrix in pair space:

          ERI_mat[pq,rs] = dot(d[:,pq], d[:,rs])

        This can be faster than the default two-step DF contraction when the number
        of contributing `rs` terms is small.
        """

        pair_ids = np.asarray(pair_ids, dtype=np.int32).ravel()
        coeff = np.asarray(coeff, dtype=np.float64).ravel()
        if pair_ids.size != coeff.size:
            raise ValueError("pair_ids and coeff must have same length")

        norb = int(self.norb)
        nops = norb * norb
        if pair_ids.size == 0:
            return np.zeros(nops, dtype=np.float64)

        eri_mat_max_bytes = int(eri_mat_max_bytes)
        if eri_mat_max_bytes > 0:
            # Prefer the explicit ERI matrix when the column count is small enough.
            # `eri_mat[pair_ids]` copies a (ncol, nops) block, so keep `ncol` modest.
            # Heuristic: when `m = len(pair_ids)` is small (<= naux), a dense-like
            # matvec in pair space is typically cheaper than the 2-step DF
            # contraction that always pays an `O(nops*naux)` cost.
            max_cols = min(int(self.naux), nops)
            use_eri_mat = pair_ids.size <= max_cols
            if use_eri_mat:
                eri_mat = self._maybe_build_eri_mat(eri_mat_max_bytes)
                if eri_mat is not None:
                    # Use row-slicing (contiguous) and symmetry (pq|rs)=(rs|pq) to avoid
                    # slow column fancy-indexing.
                    return float(half) * (coeff @ eri_mat[pair_ids])

        w = coeff @ self.l_full[pair_ids]
        return float(half) * (self.l_full @ w)

    def rr_slice_h_eff(self, occ: np.ndarray, *, half: float = 0.5, eri_mat_max_bytes: int = 0) -> np.ndarray:
        """Return term2[p,q] = half * sum_r (p q| r r) occ[r] as a (norb,norb) array."""

        norb = int(self.norb)
        occ = np.asarray(occ, dtype=np.float64).ravel()
        if occ.size != norb:
            raise ValueError("occ has wrong length")
        rr_ids = np.arange(norb, dtype=np.int32) * (norb + 1)
        out = self.contract_cols(rr_ids, occ, half=float(half), eri_mat_max_bytes=int(eri_mat_max_bytes))
        return out.reshape(norb, norb)


@dataclass(frozen=True)
class DeviceDFMOIntegrals:
    """GPU-resident active-space DF integrals.

    This is a lightweight container intended for wiring GPU-built active-space
    DF/Cholesky objects (e.g. from `gpu4pyscf`) into the GUGA CUDA matvec path.

    Arrays are expected to be CuPy ndarrays, but are typed as `Any` to avoid a
    hard CuPy dependency in the core solver.
    """

    norb: int
    l_full: Any | None  # (norb*norb, naux)
    j_ps: Any  # (norb, norb)
    pair_norm: Any | None  # (norb*norb,)
    eri_mat: Any | None  # (norb*norb, norb*norb)


def build_df_mo_integrals(
    mol,
    mo_coeff: np.ndarray,
    *,
    auxbasis: Any = "autoaux",
    filename: str | Path | None = None,
    dataname: str = "eri_mo",
    tmpdir: str | None = None,
    max_memory: int = 2000,
    verbose: int = 0,
) -> DFMOIntegrals:
    """Build MO-basis DF/Cholesky vectors for a given orbital subspace.

    By default (when `filename is None`), this uses a cached AO DF "context":
    AO-space DF/Cholesky vectors are built once per (mol, auxbasis, ...) and the
    AO->MO transform is performed in-memory (no MO HDF5 round-trip).

    Parameters
    ----------
    mo_coeff:
        AO->MO coefficient matrix with shape (nao, norb_subspace).
    filename:
        Optional HDF5 filename to store the intermediate MO 3-index tensor. If
        None, the MO tensor is built in-memory from a cached AO DF context.
    """

    mo_coeff = np.asarray(mo_coeff, dtype=np.float64)
    if mo_coeff.ndim != 2:
        raise ValueError("mo_coeff must be a 2D array (nao, norb)")

    norb = int(mo_coeff.shape[1])
    from asuka.integrals.df_context import get_df_cholesky_context  # noqa: PLC0415

    ctx = get_df_cholesky_context(
        mol,
        auxbasis=auxbasis,
        max_memory=int(max_memory),
        verbose=int(verbose),
    )
    mo_cderi_s2 = ctx.transform(mo_coeff, mo_coeff, compact=True, cache=True, max_memory=int(max_memory))

    if filename is not None:
        try:
            import h5py  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("h5py is required to write DF MO tensors to HDF5") from e

        filename = str(filename)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(str(filename), "w") as f:
            f.create_dataset(str(dataname), data=np.asarray(mo_cderi_s2, dtype=np.float64), compression="gzip")

    return DFMOIntegrals.from_mo_cderi_s2(mo_cderi_s2, norb=norb)
