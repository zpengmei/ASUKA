"""F3 contractions for CASPT2 cases A/C (OpenMolcas DELTA3).

OpenMolcas stores the active-space, Fock-contracted 4-body quantity as
`DELTA3` (`mkfg3.f`) and later scatters it into the case-A/case-C B matrices
(`mkbmat.f`).

Naively, one can form a "raw" contraction without a 4-RDM by building:

    |fc> = (sum_w epsa[w] E_ww) |c>

and evaluating:

    F3_raw(t,u,v,x,y,z) = <c| E_tu E_vx E_yz |fc>.

However, OpenMolcas' `DELTA3` corresponds to the *irreducible* unitary-group
generator `E_tuvxyzww` contracted with `epsa[w]`. In `mkfg3.f`, Molcas applies
additional delta corrections to convert product expectations into the
irreducible-generator convention. These corrections are crucial for energy-side
parity (notably the case-C/ATVX term).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.oracle import _get_epq_action_cache
from asuka.rdm.rdm123 import _STEP_TO_OCC_F64, _fill_epq_vec

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


@dataclass(frozen=True)
class CASPT2CIContext:
    """Context needed to build F3 contributions in IC-CASPT2 cases A and C.

    Carries the active-space DRT (Distinct Row Table) and the CI coefficient
    vector in the CSF basis, which are used by ``F3ContractionEngine`` to
    compute Fock-weighted 4-body quantities without an explicit 4-RDM.

    Attributes
    ----------
    drt : DRT
        Active-space GUGA Distinct Row Table (defines norb, nelec, ncsf).
    ci_csf : np.ndarray
        CI coefficient vector in the CSF basis, shape ``(ncsf,)``.
    max_memory_mb : float
        Memory cap for intermediate buffers.
    block_nops : int
        Block size for batched E_pq applications.
    """

    drt: DRT
    ci_csf: np.ndarray
    max_memory_mb: float = 4000.0
    block_nops: int = 8


class F3ContractionEngine:
    """On-demand F3 contraction engine for one CASPT2 reference state.

    Computes Fock-weighted 4-body quantities (OpenMolcas DELTA3) needed by
    cases A and C of the IC-CASPT2 B matrix, *without* building an explicit
    4-RDM.  The key identity used is:

        F3_raw(t,u,v,x,y,z) = <c| E_tu (H_diag - epsa[v]) E_vx E_yz |c>

    where ``H_diag`` is the diagonal one-electron Fock operator in the CSF
    basis and ``|c>`` is the reference CI vector.  OpenMolcas applies
    additional delta-function correction terms (from ``mkfg3.f``) to convert
    this raw contraction into the irreducible-generator DELTA3 quantity.

    The engine precomputes:
      - ``|fc> = (sum_w epsa[w] E_ww)|c>``  (Fock-weighted ket)
      - ``T1[pq] = E_pq|c>``  (all nops first-order intermediates)
      - ``bra[pq] = E_qp|c>``  (for bra contractions)

    F3 elements are evaluated lazily and cached per (y,z) pair.  Two backends
    are available for E_pq applications: sparse CSC matrices (via SciPy/Cython)
    or the oracle-based ``_fill_epq_vec`` fallback.
    """

    def __init__(self, context: CASPT2CIContext, epsa: np.ndarray):
        self.drt = context.drt
        self.norb = int(self.drt.norb)
        self.ncsf = int(self.drt.ncsf)
        self.nops = self.norb * self.norb

        self.c = np.asarray(context.ci_csf, dtype=np.float64).ravel()
        if self.c.size != self.ncsf:
            raise ValueError("ci_csf has wrong length for provided DRT")

        self.epsa = np.asarray(epsa, dtype=np.float64).ravel()
        if self.epsa.shape != (self.norb,):
            raise ValueError(f"epsa shape {self.epsa.shape} incompatible with norb={self.norb}")

        self.cache = _get_epq_action_cache(self.drt)
        self.occ = np.asarray(_STEP_TO_OCC_F64[self.cache.steps], dtype=np.float64, order="C")

        self.mats = None
        if _sp is not None and _epq_spmat_list is not None:
            self.mats = _epq_spmat_list(self.drt, self.cache)

        # Diagonal 1-el Hamiltonian in the CSF basis (MKFG3: BUFD).
        # For each CSF, this is sum_w epsa[w] * occ(csf, w).
        self.hdiag = np.asarray(self.occ @ self.epsa, dtype=np.float64, order="C")  # (ncsf,)

        # Convenience weighted ket |fc> = (sum_w epsa[w] E_ww)|c>.
        # This is useful for validating the raw transition-3RDM identity, but is not
        # the quantity OpenMolcas ultimately stores as DELTA3.
        self.fc = np.asarray(self.c * self.hdiag, dtype=np.float64, order="C")

        # T1[pq,:] = E_pq |c>,  T1_fc[pq,:] = E_pq |fc>.
        self.t1 = self._build_t1(self.c)
        self.t1_fc = self._build_t1(self.fc)

        # Bra rows: B[pq,:] = E_qp|c> so B[pq]·x = <c|E_pq|x>.
        self.bra = np.asarray(
            self.t1.reshape(self.norb, self.norb, self.ncsf).transpose(1, 0, 2).reshape(self.nops, self.ncsf),
            dtype=np.float64,
            order="C",
        )

        # Cache M_yz where M[pq,vx] = <c|E_pq E_vx E_yz|fc>.
        self._yz_cache: dict[int, np.ndarray] = {}

        # Cache R_yz where R[pq,vx] = <c|E_pq (Hdiag - epsa[v]) E_vx E_yz|c>.
        # This matches the "raw F3" construction in OpenMolcas `mkfg3.f` before the
        # final delta corrections are applied.
        self._yz_cache_f3raw: dict[int, np.ndarray] = {}

    def _fill_epq_vec(self, p: int, q: int, x: np.ndarray, out: np.ndarray) -> None:
        p = int(p)
        q = int(q)
        if p == q:
            np.multiply(self.occ[:, p], x, out=out)
            return
        if self.mats is not None:
            mat = self.mats[p * self.norb + q]
            if mat is None:
                raise AssertionError("missing E_pq sparse matrix")
            if _csc_matmul_dense_inplace_cy is not None:
                _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
                    mat.indptr, mat.indices, mat.data, x.reshape(self.ncsf, 1), out.reshape(self.ncsf, 1)
                )
            else:
                out[:] = mat.dot(x)  # type: ignore[operator]
            return
        _fill_epq_vec(self.drt, self.cache, x, p=p, q=q, out=out)

    def _build_t1(self, civec: np.ndarray) -> np.ndarray:
        t1 = np.empty((self.nops, self.ncsf), dtype=np.float64)
        for p in range(self.norb):
            for q in range(self.norb):
                pq = p * self.norb + q
                self._fill_epq_vec(p, q, civec, t1[pq])
        return np.asarray(t1, dtype=np.float64, order="C")

    def matrix_for_yz(self, y: int, z: int) -> np.ndarray:
        """Return M[pq,vx] = <c|E_pq E_vx E_yz|fc> for fixed (y,z)."""
        yz = int(y) * self.norb + int(z)
        cached = self._yz_cache.get(yz)
        if cached is not None:
            return cached

        fyz = self.t1_fc[yz]  # E_yz|fc>
        k = np.empty((self.nops, self.ncsf), dtype=np.float64)
        for v in range(self.norb):
            for x in range(self.norb):
                vx = v * self.norb + x
                self._fill_epq_vec(v, x, fyz, k[vx])

        mat = np.asarray(self.bra @ k.T, dtype=np.float64, order="C")
        self._yz_cache[yz] = mat
        return mat

    def _matrix_for_yz_f3raw(self, y: int, z: int) -> np.ndarray:
        """Return R[pq,vx] = <c|E_pq (Hdiag - epsa[v]) E_vx E_yz|c> for fixed (y,z).

        This mirrors the construction in OpenMolcas `mkfg3.f`, where after building the
        intermediate `Tau = E_vx E_yz |c>`, Molcas multiplies it elementwise by
        `(Hdiag - epsa[v])` before contracting with the `E_ut|c>` buffer.
        """

        yz = int(y) * self.norb + int(z)
        cached = self._yz_cache_f3raw.get(yz)
        if cached is not None:
            return cached

        yz_vec = self.t1[yz]  # E_yz|c>
        k = np.empty((self.nops, self.ncsf), dtype=np.float64)

        for v in range(self.norb):
            scale = self.hdiag - self.epsa[v]
            for x in range(self.norb):
                vx = v * self.norb + x
                self._fill_epq_vec(v, x, yz_vec, k[vx])
                np.multiply(k[vx], scale, out=k[vx])

        mat = np.asarray(self.bra @ k.T, dtype=np.float64, order="C")
        self._yz_cache_f3raw[yz] = mat
        return mat

    def f3_raw_mkfg3(self, t: int, u: int, v: int, x: int, y: int, z: int) -> float:
        """Raw `mkfg3.f` contraction before DELTA3 corrections."""
        mat = self._matrix_for_yz_f3raw(y, z)
        tu = int(t) * self.norb + int(u)
        vx = int(v) * self.norb + int(x)
        return float(mat[tu, vx])

    def f3_molcas(
        self,
        t: int,
        u: int,
        v: int,
        x: int,
        y: int,
        z: int,
        dm2: np.ndarray,
        dm3: np.ndarray,
        fd: np.ndarray,
        fp: np.ndarray,
    ) -> float:
        """OpenMolcas `DELTA3`: sum_w epsa[w] * <E_tuvxyzww>.

        This applies the correction rules from `OpenMolcas/src/caspt2/mkfg3.f`
        (see the `Correction to G3 ... Similar for F3 values.` block) to the raw
        transition contraction produced by the `|fc>` trick.

        Parameters
        ----------
        (t,u,v,x,y,z):
            Active indices, corresponding to the three E-operator pairs
            (tu),(vx),(yz) in `mkfg3.f`.
        dm2, dm3:
            Molcas-convention active `G2` and `G3` tensors (irreducible generators).
        fd, fp:
            Fock-contracted intermediates built from (dm2, dm3):
              - fd[t,z] = F1(t,z)
              - fp[p,q,r,s] = 0.5 * F2(p,q,r,s)
        """

        f3 = self.f3_raw_mkfg3(t, u, v, x, y, z)
        epsa = self.epsa

        # MKFG3 correction terms (OpenMolcas/src/caspt2/mkfg3.f).
        if int(y) == int(x):
            f3 -= (2.0 * fp[t, u, v, z] + epsa[u] * dm2[t, u, v, z])
            if int(v) == int(u):
                f3 -= fd[t, z]
        if int(v) == int(u):
            f3 -= (2.0 * fp[t, x, y, z] + epsa[y] * dm2[t, x, y, z])
        if int(y) == int(u):
            f3 -= (2.0 * fp[v, x, t, z] + epsa[u] * dm2[v, x, t, z])
        f3 -= (epsa[u] + epsa[y]) * dm3[t, u, v, x, y, z]
        return float(f3)

    def f3_case_a(
        self,
        v: int,
        u: int,
        x: int,
        t: int,
        y: int,
        z: int,
        dm2: np.ndarray,
        dm3: np.ndarray,
        fd: np.ndarray,
        fp: np.ndarray,
    ) -> float:
        """Return DELTA3 element used by MKBA: F3[v,u,x,t,y,z]."""
        return self.f3_molcas(v, u, x, t, y, z, dm2, dm3, fd, fp)

    def f3_case_c(
        self,
        v: int,
        u: int,
        t: int,
        x: int,
        y: int,
        z: int,
        dm2: np.ndarray,
        dm3: np.ndarray,
        fd: np.ndarray,
        fp: np.ndarray,
    ) -> float:
        """Return DELTA3 element used by MKBC: F3[v,u,t,x,y,z]."""
        return self.f3_molcas(v, u, t, x, y, z, dm2, dm3, fd, fp)
