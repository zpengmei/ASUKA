from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.oracle import _csr_for_epq, _get_epq_action_cache

_STEP_TO_OCC_F64 = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float64)  # E,U,L,D

try:  # optional Cython in-place CSC @ dense kernels
    from asuka._epq_cy import (  # type: ignore[import-not-found]
        csc_matmul_dense_inplace_cy as _csc_matmul_dense_inplace_cy,
    )
except Exception:  # pragma: no cover
    _csc_matmul_dense_inplace_cy = None


@dataclass(frozen=True)
class SpinFreeCASSCFStateDet:
    """Spin-free CASSCF-like state for RASSI-style SI (determinant CI).

    Notes
    -----
    - ``mo_core`` and ``mo_act`` are AO->MO coefficient matrices for the inactive
      (doubly occupied) and active orbital subspaces, respectively.
    - ``ci`` is the determinant-based CI vector for the *active* space only,
      in the standard PySCF FCI layout (alpha-string index × beta-string index).
    """

    twos: int
    mo_core: np.ndarray  # (nao,ncore)
    mo_act: np.ndarray  # (nao,ncas)
    ci: np.ndarray  # (na_det_a, na_det_b)
    nelecas: int | tuple[int, int]


@dataclass(frozen=True)
class SpinFreeCASSCFStateCSF:
    """Spin-free CASSCF-like state for RASSI-style SI (GUGA/CSF CI).

    Notes
    -----
    - ``drt`` and ``ci`` define the active-space spin-adapted CSF expansion.
    - ``mo_core`` and ``mo_act`` are AO->MO coefficient matrices for the inactive
      (doubly occupied) and active orbital subspaces, respectively.
    """

    twos: int
    drt: DRT
    ci: np.ndarray  # (ncsf,)
    mo_core: np.ndarray  # (nao,ncore)
    mo_act: np.ndarray  # (nao,ncas)
    nelecas: int | tuple[int, int]

    def __post_init__(self) -> None:
        if int(self.twos) != int(self.drt.twos_target):
            raise ValueError("twos must match drt.twos_target")
        ci = np.asarray(self.ci)
        if ci.ndim != 1:
            raise ValueError("ci must be a 1D array")
        if int(ci.size) != int(self.drt.ncsf):
            raise ValueError("ci has wrong length for drt")
        mo_act = np.asarray(self.mo_act)
        if mo_act.ndim != 2 or int(mo_act.shape[1]) != int(self.drt.norb):
            raise ValueError("mo_act must have shape (nao, drt.norb)")


@dataclass(frozen=True)
class SpinFreeRASSIResult:
    """Spin-free RASSI result for a non-orthogonal state basis."""

    energies: np.ndarray  # (nroot,)
    twos: np.ndarray  # (nroot,), doubled spin quantum numbers (2S)
    mixing: np.ndarray  # (nstate,nroot), columns are orthonormalized eigenstates in the input basis
    overlap: np.ndarray  # (nstate,nstate)
    hamiltonian: np.ndarray  # (nstate,nstate)


@dataclass(frozen=True)
class SOCRASSIBiorthCSFResult:
    """SOC-RASSI result from CSF/GUGA wavefunctions with non-orthogonal orbitals."""

    spinfree: SpinFreeRASSIResult
    gm_original: np.ndarray  # (nstate,nstate,3), reduced SOC couplings between original states
    gm_rassi: np.ndarray  # (nroot,nroot,3), reduced SOC couplings in the orthonormalized RASSI basis
    so_energies: np.ndarray  # (nss,)
    so_vectors: np.ndarray  # (nss,nss), spin-component basis
    so_basis: list[tuple[int, int]]  # (rassi_state_index, tm=2M)


def _epq_matvec(
    drt: DRT,
    cache,
    x: np.ndarray,
    *,
    p: int,
    q: int,
    out: np.ndarray,
) -> None:
    """Compute out[:] = E_pq |x> in the DRT basis."""

    p = int(p)
    q = int(q)
    if p == q:
        out[:] = _STEP_TO_OCC_F64[cache.steps[:, p]] * x
        return

    csr = _csr_for_epq(cache, drt, p, q)
    if _csc_matmul_dense_inplace_cy is not None:
        x_col = x.reshape(int(x.size), 1)
        out_col = out.reshape(int(out.size), 1)
        _csc_matmul_dense_inplace_cy(  # type: ignore[attr-defined]
            csr.indptr,
            csr.indices,
            csr.data,
            x_col,
            out_col,
        )
        return

    out.fill(0.0)
    indptr = csr.indptr
    indices = csr.indices
    data = csr.data
    for j in range(int(x.size)):
        xj = float(x[j])
        if xj == 0.0:
            continue
        start = int(indptr[j])
        end = int(indptr[j + 1])
        if start == end:
            continue
        out[indices[start:end]] += data[start:end] * xj


def _sigma1_add(
    drt: DRT,
    cache,
    p: int,
    q: int,
    coeff: float,
    x: np.ndarray,
    out: np.ndarray,
    work: np.ndarray,
) -> None:
    """Accumulate out += coeff * (E_pq |x>) using a work buffer."""

    coeff = float(coeff)
    if coeff == 0.0:
        return
    p = int(p)
    q = int(q)
    if p == q:
        out += coeff * _STEP_TO_OCC_F64[cache.steps[:, p]] * x
        return
    _epq_matvec(drt, cache, x, p=p, q=q, out=work)
    out += coeff * work


def transform_csf_ci_for_orbital_transform(
    drt: DRT,
    ci: np.ndarray,
    tra: np.ndarray,
    *,
    tol: float = 1e-14,
) -> np.ndarray:
    """Transform a CSF CI vector under a general one-particle transformation.

    This mirrors OpenMolcas RASSI's CITRA/SSOTRA logic for a single symmetry block
    (i.e. C1): apply a sequence of single-orbital transformations using the
    spin-summed generator actions ``E_pq`` in the DRT basis.

    Parameters
    ----------
    drt
        Active-space DRT defining the CSF basis.
    ci
        CI vector in the DRT ordering (length ``drt.ncsf``).
    tra
        Active-orbital transformation matrix with shape ``(norb,norb)``. For RASSI
        biorthonormalization, this is typically the inverse of the orbital
        transformation applied to the MO coefficients.
    tol
        Small-coefficient threshold mirroring OpenMolcas' 1e-14 cut.
    """

    norb = int(drt.norb)
    ncsf = int(drt.ncsf)
    ci0 = np.asarray(ci, dtype=np.float64).ravel()
    if ci0.size != ncsf:
        raise ValueError("ci has wrong length for drt")
    tra = np.asarray(tra, dtype=np.float64)
    if tra.shape != (norb, norb):
        raise ValueError("tra must have shape (drt.norb, drt.norb)")

    tol = float(tol)
    if tol < 0.0:
        raise ValueError("tol must be >= 0")

    cache = _get_epq_action_cache(drt)
    ci_out = np.array(ci0, dtype=np.float64, copy=True)
    tmp = np.zeros(ncsf, dtype=np.float64)
    work = np.empty(ncsf, dtype=np.float64)

    for k in range(norb):
        tmp.fill(0.0)
        for p in range(norb):
            cpk = float(tra[p, k])
            if p == k:
                cpk -= 1.0
            x = 0.5 * cpk
            if abs(x) < tol:
                continue
            _sigma1_add(drt, cache, p, k, x, ci_out, tmp, work)

        ckk = float(tra[k, k])
        ci_out += (3.0 - ckk) * tmp

        for p in range(norb):
            cpk = float(tra[p, k])
            if p == k:
                cpk -= 1.0
            if abs(cpk) < tol:
                continue
            _sigma1_add(drt, cache, p, k, cpk, tmp, ci_out, work)

    return ci_out


def _unpack_nelecas(nelecas: int | tuple[int, int], *, twos: int) -> tuple[int, int]:
    if isinstance(nelecas, (tuple, list, np.ndarray)):
        if len(nelecas) != 2:
            raise ValueError("nelecas must be an int or a (neleca,nelecb) pair")
        neleca = int(nelecas[0])
        nelecb = int(nelecas[1])
        if neleca < 0 or nelecb < 0:
            raise ValueError("nelecas must be non-negative")
        return neleca, nelecb

    nelec = int(nelecas)
    twos = int(twos)
    if (nelec + twos) & 1:
        raise ValueError("nelecas and twos parity mismatch")
    neleca = (nelec + twos) // 2
    nelecb = nelec - neleca
    if neleca < 0 or nelecb < 0:
        raise ValueError("invalid nelecas for given twos")
    return int(neleca), int(nelecb)


def _biorth_svd(
    overlap: np.ndarray,
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_bra, x_ket) such that x_bra.T @ overlap @ x_ket == I.

    The returned matrices define biorthonormal orbitals:

      C_bra' = C_bra @ x_bra
      C_ket' = C_ket @ x_ket

    assuming ``overlap = C_bra.T @ S_ao @ C_ket`` with ``C_bra`` and ``C_ket`` individually
    orthonormal in the AO metric.
    """

    o = np.asarray(overlap, dtype=np.float64)
    if o.ndim != 2 or o.shape[0] != o.shape[1]:
        raise ValueError("overlap must be a square 2D array")
    n = int(o.shape[0])
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.float64)

    u, s, vt = np.linalg.svd(o, full_matrices=False)
    s = np.asarray(s, dtype=np.float64)
    s_max = float(np.max(s))
    if not math.isfinite(s_max) or s_max <= 0.0:
        raise np.linalg.LinAlgError("singular orbital overlap matrix")
    tol = float(tol)
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    s_min = float(np.min(s))
    if s_min <= tol * s_max:
        raise np.linalg.LinAlgError(
            f"orbital overlap matrix is numerically singular: min={s_min:.3e}, max={s_max:.3e}, tol={tol:.3e}"
        )

    inv_sqrt = 1.0 / np.sqrt(s)
    x_bra = u * inv_sqrt[None, :]
    x_ket = vt.T * inv_sqrt[None, :]
    return np.asarray(x_bra, dtype=np.float64), np.asarray(x_ket, dtype=np.float64)


def _lu_nopivot(a: np.ndarray, *, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Pivot-free LU factorization A = L U (Doolittle), with diag(L)=1.

    This is used to build an *upper-triangular* biorthogonalization that preserves
    the (core | active) orbital ordering (core columns do not mix with active).
    """

    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be a square 2D array")
    n = int(a.shape[0])
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.float64)

    tol = float(tol)
    if tol <= 0.0:
        raise ValueError("tol must be positive")

    max_abs = float(np.max(np.abs(a)))
    thresh = tol * max_abs

    l = np.eye(n, dtype=np.float64)
    u = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        if k == 0:
            u[k, k:] = a[k, k:]
        else:
            u[k, k:] = a[k, k:] - l[k, :k] @ u[:k, k:]
        pivot = float(u[k, k])
        if not math.isfinite(pivot) or abs(pivot) <= max(thresh, 1e-300):
            raise np.linalg.LinAlgError(
                f"pivot-free LU failed (near-singular overlap): k={k}, pivot={pivot:.3e}, thresh={thresh:.3e}"
            )
        if k + 1 < n:
            if k == 0:
                l[k + 1 :, k] = a[k + 1 :, k] / pivot
            else:
                l[k + 1 :, k] = (a[k + 1 :, k] - l[k + 1 :, :k] @ u[:k, k]) / pivot
    return l, u


def _biorth_mos(
    mo_bra: np.ndarray,
    mo_ket: np.ndarray,
    *,
    s_ao: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mo_bra = np.asarray(mo_bra, dtype=np.float64)
    mo_ket = np.asarray(mo_ket, dtype=np.float64)
    if mo_bra.ndim != 2 or mo_ket.ndim != 2 or mo_bra.shape != mo_ket.shape:
        raise ValueError("mo_bra and mo_ket must be 2D arrays with the same shape")
    s_ao = np.asarray(s_ao, dtype=np.float64)
    if s_ao.ndim != 2 or s_ao.shape[0] != s_ao.shape[1] or s_ao.shape[0] != mo_bra.shape[0]:
        raise ValueError("s_ao must be a square AO overlap matrix compatible with mo_bra/mo_ket")

    o = mo_bra.T @ s_ao @ mo_ket
    x_bra, x_ket = _biorth_svd(o, tol=float(tol))
    mo_bra_bi = mo_bra @ x_bra
    mo_ket_bi = mo_ket @ x_ket
    return mo_bra_bi, mo_ket_bi, x_bra, x_ket


def spinfree_rassi_h_s_pair_biorth(
    bra: SpinFreeCASSCFStateDet,
    ket: SpinFreeCASSCFStateDet,
    *,
    s_ao: np.ndarray,
    h1_ao: np.ndarray,
    eri_ao: np.ndarray,
    e_nuc: float,
    biorth_tol: float = 1e-12,
) -> tuple[float, float]:
    """Compute (H_ij, S_ij) for two CASSCF-like states using biorthonormal orbitals.

    This follows the OpenMolcas RASSI idea:
    - biorthonormalize inactive (core) and active orbital subspaces (no mixing between them),
    - transform the CI vectors to the biorthonormal active orbitals,
    - evaluate overlap and Hamiltonian matrix elements using transition RDMs.

    Limitations
    -----------
    - Requires determinant CI vectors (PySCF FCI layout).
    - Assumes inactive and active subspaces are well-defined and do not mix.
    """

    if int(bra.twos) != int(ket.twos):
        # Spin-free Hamiltonian and overlap do not couple different spins.
        return 0.0, 0.0
    neleca, nelecb = _unpack_nelecas(bra.nelecas, twos=int(bra.twos))
    if (neleca, nelecb) != _unpack_nelecas(ket.nelecas, twos=int(ket.twos)):
        return 0.0, 0.0

    mo_core_bra = np.asarray(bra.mo_core, dtype=np.float64)
    mo_core_ket = np.asarray(ket.mo_core, dtype=np.float64)
    mo_act_bra = np.asarray(bra.mo_act, dtype=np.float64)
    mo_act_ket = np.asarray(ket.mo_act, dtype=np.float64)

    if mo_core_bra.shape != mo_core_ket.shape:
        raise ValueError("bra/ket mo_core shapes mismatch")
    if mo_act_bra.shape != mo_act_ket.shape:
        raise ValueError("bra/ket mo_act shapes mismatch")
    if mo_core_bra.shape[0] != mo_act_bra.shape[0]:
        raise ValueError("mo_core and mo_act must have the same number of rows (AOs)")

    ncore = int(mo_core_bra.shape[1])
    ncas = int(mo_act_bra.shape[1])

    if ncas <= 0:
        raise ValueError("ncas must be positive")
    if neleca + nelecb < 0:
        raise ValueError("invalid nelecas")

    tol = float(biorth_tol)

    # Biorthonormalize the *combined occupied* space (core + active) so that the
    # determinant overlap becomes a pure dot-product in the biorth basis. We use a
    # pivot-free LU-based biorthogonalization to preserve the core|active ordering.
    mo_occ_bra = np.hstack([mo_core_bra, mo_act_bra])
    mo_occ_ket = np.hstack([mo_core_ket, mo_act_ket])
    nocc = int(mo_occ_bra.shape[1])
    if nocc != ncore + ncas:
        raise RuntimeError("internal error: nocc mismatch")

    o_occ = mo_occ_bra.T @ s_ao @ mo_occ_ket
    l, u = _lu_nopivot(o_occ, tol=tol)
    inv_l = np.linalg.solve(l, np.eye(nocc, dtype=np.float64))
    inv_u = np.linalg.solve(u, np.eye(nocc, dtype=np.float64))
    x_occ_bra = inv_l.T
    x_occ_ket = inv_u

    mo_occ_bra_bi = mo_occ_bra @ x_occ_bra
    mo_occ_ket_bi = mo_occ_ket @ x_occ_ket
    mo_core_bra_bi = mo_occ_bra_bi[:, :ncore]
    mo_core_ket_bi = mo_occ_ket_bi[:, :ncore]
    mo_act_bra_bi = mo_occ_bra_bi[:, ncore:]
    mo_act_ket_bi = mo_occ_ket_bi[:, ncore:]

    x_core_bra = x_occ_bra[:ncore, :ncore]
    x_core_ket = x_occ_ket[:ncore, :ncore]
    x_act_bra = x_occ_bra[ncore:, ncore:]
    x_act_ket = x_occ_ket[ncore:, ncore:]

    if ncore:
        det_core_bra = float(np.linalg.det(x_core_bra))
        det_core_ket = float(np.linalg.det(x_core_ket))
        core_scale_bra = det_core_bra ** (-2)
        core_scale_ket = det_core_ket ** (-2)
    else:
        core_scale_bra = 1.0
        core_scale_ket = 1.0

    # Transform determinant CI to the biorthonormal active orbitals.
    try:  # pragma: no cover
        from pyscf.fci import addons as fci_addons
        from pyscf.fci import direct_spin1
    except Exception as e:  # pragma: no cover
        raise RuntimeError("spinfree_rassi_h_s_pair_biorth requires PySCF (pyscf.fci)") from e

    ci_bra = np.asarray(bra.ci, dtype=np.float64, order="C")
    ci_ket = np.asarray(ket.ci, dtype=np.float64, order="C")

    # For a general (non-unitary) orbital transformation, the CI coefficient update is
    # contravariant in the one-particle transformation. In the determinant basis, this
    # corresponds to using the inverse-transpose transformation matrix.
    invt_act_bra = np.linalg.inv(x_act_bra).T
    invt_act_ket = np.linalg.inv(x_act_ket).T
    ci_bra_bi = fci_addons.transform_ci_for_orbital_rotation(ci_bra, ncas, (neleca, nelecb), invt_act_bra)
    ci_ket_bi = fci_addons.transform_ci_for_orbital_rotation(ci_ket, ncas, (neleca, nelecb), invt_act_ket)

    ci_bra_bi = np.asarray(ci_bra_bi, dtype=np.float64, order="C") * float(core_scale_bra)
    ci_ket_bi = np.asarray(ci_ket_bi, dtype=np.float64, order="C") * float(core_scale_ket)

    s_ij = float(np.dot(ci_bra_bi.ravel(), ci_ket_bi.ravel()))

    dm1, dm2 = direct_spin1.trans_rdm12(ci_bra_bi, ci_ket_bi, ncas, (neleca, nelecb), reorder=True)

    # Mixed AO->(core+active) MO integrals in the biorthonormal bases.
    s_ao = np.asarray(s_ao, dtype=np.float64)
    h1_ao = np.asarray(h1_ao, dtype=np.float64)
    eri_ao = np.asarray(eri_ao, dtype=np.float64)
    if h1_ao.shape != s_ao.shape:
        raise ValueError("h1_ao must have the same shape as s_ao")
    if eri_ao.ndim != 4 or eri_ao.shape[0] != s_ao.shape[0] or eri_ao.shape[1] != s_ao.shape[0]:
        raise ValueError("eri_ao must have shape (nao,nao,nao,nao) compatible with s_ao")

    h_occ = mo_occ_bra_bi.T @ h1_ao @ mo_occ_ket_bi
    # Two-electron integrals for off-diagonal (bra|H|ket) matrix elements must interleave
    # bra/ket orbital coefficients per electron coordinate:
    #   (p q | r s) = ∫ φ_p^bra(1) φ_q^ket(1) r12^-1 φ_r^bra(2) φ_s^ket(2) d1 d2
    eri_occ = np.einsum(
        "ap,bq,abcd,cr,ds->pqrs",
        mo_occ_bra_bi,
        mo_occ_ket_bi,
        eri_ao,
        mo_occ_bra_bi,
        mo_occ_ket_bi,
        optimize=True,
    )

    # Core energy constant term.
    ecore = float(e_nuc)
    if ncore:
        h_cc = h_occ[:ncore, :ncore]
        eri_cccc = eri_occ[:ncore, :ncore, :ncore, :ncore]
        e1_core = 2.0 * float(np.trace(h_cc))
        e2_core = 2.0 * float(np.einsum("iijj->", eri_cccc, optimize=True)) - float(
            np.einsum("ijji->", eri_cccc, optimize=True)
        )
        ecore += e1_core + e2_core

    # Effective one-electron integrals in the active space (includes core contributions).
    h_act = h_occ[ncore:, ncore:]
    if h_act.shape != (ncas, ncas):
        raise RuntimeError("internal error: h_act shape mismatch")
    h_eff = h_act.copy()
    if ncore:
        eri_accc = eri_occ[ncore:, ncore:, :ncore, :ncore]
        eri_accc2 = eri_occ[ncore:, :ncore, :ncore, ncore:]
        # h_eff[p,q] += Σ_i [2 (p q | i i) - (p i | i q)]
        h_eff += 2.0 * np.einsum("pqii->pq", eri_accc, optimize=True) - np.einsum("piiq->pq", eri_accc2, optimize=True)

    eri_act = eri_occ[ncore:, ncore:, ncore:, ncore:]
    if eri_act.shape != (ncas, ncas, ncas, ncas):
        raise RuntimeError("internal error: eri_act shape mismatch")

    hij_act = float(np.einsum("pq,qp->", h_eff, dm1, optimize=True) + 0.5 * np.einsum("pqrs,pqrs->", eri_act, dm2, optimize=True))
    hij = ecore * s_ij + hij_act

    return float(hij), float(s_ij)


def spinfree_rassi_h_s_pair_biorth_csf(
    bra: SpinFreeCASSCFStateCSF,
    ket: SpinFreeCASSCFStateCSF,
    *,
    s_ao: np.ndarray,
    h1_ao: np.ndarray,
    eri_ao: np.ndarray,
    e_nuc: float,
    biorth_tol: float = 1e-12,
    ci_transform_tol: float = 1e-14,
    rdm_max_memory_mb: float = 4000.0,
    rdm_block_nops: int = 8,
) -> tuple[float, float]:
    """Compute (H_ij, S_ij) for two CSF-based states using biorthonormal orbitals."""

    if int(bra.twos) != int(ket.twos):
        return 0.0, 0.0

    neleca, nelecb = _unpack_nelecas(bra.nelecas, twos=int(bra.twos))
    if (neleca, nelecb) != _unpack_nelecas(ket.nelecas, twos=int(ket.twos)):
        return 0.0, 0.0

    if bra.drt.norb != ket.drt.norb or bra.drt.nelec != ket.drt.nelec or bra.drt.twos_target != ket.drt.twos_target:
        raise ValueError("bra/ket DRT mismatch")

    ncas = int(bra.drt.norb)
    if ncas <= 0:
        raise ValueError("ncas must be positive")

    mo_core_bra = np.asarray(bra.mo_core, dtype=np.float64)
    mo_core_ket = np.asarray(ket.mo_core, dtype=np.float64)
    mo_act_bra = np.asarray(bra.mo_act, dtype=np.float64)
    mo_act_ket = np.asarray(ket.mo_act, dtype=np.float64)

    if mo_core_bra.shape != mo_core_ket.shape:
        raise ValueError("bra/ket mo_core shapes mismatch")
    if mo_act_bra.shape != mo_act_ket.shape:
        raise ValueError("bra/ket mo_act shapes mismatch")
    if mo_act_bra.shape[1] != ncas:
        raise ValueError("mo_act must have ncas columns matching drt.norb")
    if mo_core_bra.shape[0] != mo_act_bra.shape[0]:
        raise ValueError("mo_core and mo_act must have the same number of rows (AOs)")

    ncore = int(mo_core_bra.shape[1])

    tol = float(biorth_tol)

    # Biorthonormalize combined occupied space (core + active), preserving core|active ordering.
    mo_occ_bra = np.hstack([mo_core_bra, mo_act_bra])
    mo_occ_ket = np.hstack([mo_core_ket, mo_act_ket])
    nocc = int(mo_occ_bra.shape[1])
    if nocc != ncore + ncas:
        raise RuntimeError("internal error: nocc mismatch")

    o_occ = mo_occ_bra.T @ s_ao @ mo_occ_ket
    l, u = _lu_nopivot(o_occ, tol=tol)
    inv_l = np.linalg.solve(l, np.eye(nocc, dtype=np.float64))
    inv_u = np.linalg.solve(u, np.eye(nocc, dtype=np.float64))
    x_occ_bra = inv_l.T
    x_occ_ket = inv_u

    mo_occ_bra_bi = mo_occ_bra @ x_occ_bra
    mo_occ_ket_bi = mo_occ_ket @ x_occ_ket

    x_core_bra = x_occ_bra[:ncore, :ncore]
    x_core_ket = x_occ_ket[:ncore, :ncore]
    x_act_bra = x_occ_bra[ncore:, ncore:]
    x_act_ket = x_occ_ket[ncore:, ncore:]

    if ncore:
        det_core_bra = float(np.linalg.det(x_core_bra))
        det_core_ket = float(np.linalg.det(x_core_ket))
        core_scale_bra = det_core_bra ** (-2)
        core_scale_ket = det_core_ket ** (-2)
    else:
        core_scale_bra = 1.0
        core_scale_ket = 1.0

    tra_act_bra = np.linalg.inv(x_act_bra)
    tra_act_ket = np.linalg.inv(x_act_ket)

    ci_bra_bi = transform_csf_ci_for_orbital_transform(bra.drt, bra.ci, tra_act_bra, tol=float(ci_transform_tol))
    ci_ket_bi = transform_csf_ci_for_orbital_transform(ket.drt, ket.ci, tra_act_ket, tol=float(ci_transform_tol))
    ci_bra_bi = np.asarray(ci_bra_bi, dtype=np.float64, order="C") * float(core_scale_bra)
    ci_ket_bi = np.asarray(ci_ket_bi, dtype=np.float64, order="C") * float(core_scale_ket)

    s_ij = float(np.dot(ci_bra_bi, ci_ket_bi))

    from asuka.rdm.stream import trans_rdm12_streaming

    dm1, dm2 = trans_rdm12_streaming(
        bra.drt,
        ci_bra_bi,
        ci_ket_bi,
        max_memory_mb=float(rdm_max_memory_mb),
        block_nops=int(rdm_block_nops),
        reorder=True,
    )

    # Mixed AO->(core+active) MO integrals in the biorthonormal bases.
    s_ao = np.asarray(s_ao, dtype=np.float64)
    h1_ao = np.asarray(h1_ao, dtype=np.float64)
    eri_ao = np.asarray(eri_ao, dtype=np.float64)
    if h1_ao.shape != s_ao.shape:
        raise ValueError("h1_ao must have the same shape as s_ao")
    if eri_ao.ndim != 4 or eri_ao.shape[0] != s_ao.shape[0] or eri_ao.shape[1] != s_ao.shape[0]:
        raise ValueError("eri_ao must have shape (nao,nao,nao,nao) compatible with s_ao")

    h_occ = mo_occ_bra_bi.T @ h1_ao @ mo_occ_ket_bi
    eri_occ = np.einsum(
        "ap,bq,abcd,cr,ds->pqrs",
        mo_occ_bra_bi,
        mo_occ_ket_bi,
        eri_ao,
        mo_occ_bra_bi,
        mo_occ_ket_bi,
        optimize=True,
    )

    # Core energy constant term.
    ecore = float(e_nuc)
    if ncore:
        h_cc = h_occ[:ncore, :ncore]
        eri_cccc = eri_occ[:ncore, :ncore, :ncore, :ncore]
        e1_core = 2.0 * float(np.trace(h_cc))
        e2_core = 2.0 * float(np.einsum("iijj->", eri_cccc, optimize=True)) - float(
            np.einsum("ijji->", eri_cccc, optimize=True)
        )
        ecore += e1_core + e2_core

    # Effective one-electron integrals in the active space (includes core contributions).
    h_act = h_occ[ncore:, ncore:]
    if h_act.shape != (ncas, ncas):
        raise RuntimeError("internal error: h_act shape mismatch")
    h_eff = h_act.copy()
    if ncore:
        eri_accc = eri_occ[ncore:, ncore:, :ncore, :ncore]
        eri_accc2 = eri_occ[ncore:, :ncore, :ncore, ncore:]
        h_eff += 2.0 * np.einsum("pqii->pq", eri_accc, optimize=True) - np.einsum("piiq->pq", eri_accc2, optimize=True)

    eri_act = eri_occ[ncore:, ncore:, ncore:, ncore:]
    if eri_act.shape != (ncas, ncas, ncas, ncas):
        raise RuntimeError("internal error: eri_act shape mismatch")

    hij_act = float(
        np.einsum("pq,qp->", h_eff, dm1, optimize=True) + 0.5 * np.einsum("pqrs,pqrs->", eri_act, dm2, optimize=True)
    )
    hij = ecore * s_ij + hij_act
    return float(hij), float(s_ij)


def build_spinfree_rassi_matrices_biorth(
    states: list[SpinFreeCASSCFStateDet],
    *,
    s_ao: np.ndarray,
    h1_ao: np.ndarray,
    eri_ao: np.ndarray,
    e_nuc: float,
    biorth_tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (H,S) in the basis of non-orthogonal input states."""

    if not states:
        raise ValueError("states is empty")

    n = int(len(states))
    h = np.zeros((n, n), dtype=np.float64)
    s = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1):
            hij, sij = spinfree_rassi_h_s_pair_biorth(
                states[i],
                states[j],
                s_ao=s_ao,
                h1_ao=h1_ao,
                eri_ao=eri_ao,
                e_nuc=float(e_nuc),
                biorth_tol=float(biorth_tol),
            )
            h[i, j] = hij
            h[j, i] = hij
            s[i, j] = sij
            s[j, i] = sij

    # Numerical cleanup.
    h = 0.5 * (h + h.T)
    s = 0.5 * (s + s.T)
    return h, s


def build_spinfree_rassi_matrices_biorth_csf(
    states: list[SpinFreeCASSCFStateCSF],
    *,
    s_ao: np.ndarray,
    h1_ao: np.ndarray,
    eri_ao: np.ndarray,
    e_nuc: float,
    biorth_tol: float = 1e-12,
    ci_transform_tol: float = 1e-14,
    rdm_max_memory_mb: float = 4000.0,
    rdm_block_nops: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (H,S) in the basis of non-orthogonal input CSF states."""

    if not states:
        raise ValueError("states is empty")

    n = int(len(states))
    h = np.zeros((n, n), dtype=np.float64)
    s = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1):
            hij, sij = spinfree_rassi_h_s_pair_biorth_csf(
                states[i],
                states[j],
                s_ao=s_ao,
                h1_ao=h1_ao,
                eri_ao=eri_ao,
                e_nuc=float(e_nuc),
                biorth_tol=float(biorth_tol),
                ci_transform_tol=float(ci_transform_tol),
                rdm_max_memory_mb=float(rdm_max_memory_mb),
                rdm_block_nops=int(rdm_block_nops),
            )
            h[i, j] = hij
            h[j, i] = hij
            s[i, j] = sij
            s[j, i] = sij

    h = 0.5 * (h + h.T)
    s = 0.5 * (s + s.T)
    return h, s


def soc_rassi_gm_pair_biorth_csf(
    bra: SpinFreeCASSCFStateCSF,
    ket: SpinFreeCASSCFStateCSF,
    *,
    s_ao: np.ndarray,
    hso_xyz_ao: np.ndarray | None = None,
    hso_m_ao: np.ndarray | None = None,
    biorth_tol: float = 1e-12,
    ci_transform_tol: float = 1e-14,
    trdm_block_nops: int = 8,
) -> np.ndarray:
    """Compute reduced SOC couplings G_m(IJ) between two CSF states with non-orthogonal orbitals.

    The coupling is computed in a RASSI-like biorthonormal occupied basis:
    - Biorthonormalize the combined occupied space (core+active) preserving core|active ordering.
    - Transform each state's CSF CI to the biorthonormal active orbitals (Molcas CITRA/SSOTRA analogue).
    - Build the triplet transition 1-RDM u[p,q] = <bra||T^(1)_{q p}||ket> (qp convention).
    - Contract with mixed-basis SOC integrals h_m[m,p,q] = <p(bra)|h_m|q(ket)>:

        G_m(IJ) = Σ_{p,q} h_m[m,p,q] * u[p,q]

    Parameters
    ----------
    bra, ket
        Spin-free CSF/GUGA states. `drt.norb` and `drt.nelec` must match; `twos` may differ.
    s_ao
        AO overlap matrix.
    hso_xyz_ao, hso_m_ao
        AO one-electron SOC integrals. Provide either Cartesian components (x,y,z) in `hso_xyz_ao`
        or spherical components (m=-1,0,+1) in `hso_m_ao`.
    """

    if hso_xyz_ao is None and hso_m_ao is None:
        raise ValueError("must provide either hso_xyz_ao or hso_m_ao")

    if bra.drt.norb != ket.drt.norb or bra.drt.nelec != ket.drt.nelec:
        raise ValueError("bra/ket DRT mismatch (norb/nelec must match)")

    ncas = int(bra.drt.norb)
    if ncas <= 0:
        raise ValueError("ncas must be positive")

    mo_core_bra = np.asarray(bra.mo_core, dtype=np.float64)
    mo_core_ket = np.asarray(ket.mo_core, dtype=np.float64)
    mo_act_bra = np.asarray(bra.mo_act, dtype=np.float64)
    mo_act_ket = np.asarray(ket.mo_act, dtype=np.float64)

    if mo_core_bra.shape != mo_core_ket.shape:
        raise ValueError("bra/ket mo_core shapes mismatch")
    if mo_act_bra.shape != mo_act_ket.shape:
        raise ValueError("bra/ket mo_act shapes mismatch")
    if int(mo_act_bra.shape[1]) != ncas:
        raise ValueError("mo_act must have ncas columns matching drt.norb")
    if mo_core_bra.shape[0] != mo_act_bra.shape[0]:
        raise ValueError("mo_core and mo_act must have the same number of rows (AOs)")

    ncore = int(mo_core_bra.shape[1])
    nao = int(mo_core_bra.shape[0])

    s_ao = np.asarray(s_ao, dtype=np.float64)
    if s_ao.shape != (nao, nao):
        raise ValueError("s_ao must have shape (nao,nao) compatible with mo_core/mo_act")

    if hso_m_ao is not None:
        h_m_ao = np.asarray(hso_m_ao, dtype=np.complex128)
    else:
        from asuka.soc.si import soc_xyz_to_spherical  # noqa: PLC0415

        h_m_ao = soc_xyz_to_spherical(np.asarray(hso_xyz_ao, dtype=np.complex128))
    if h_m_ao.ndim != 3 or h_m_ao.shape[0] != 3 or h_m_ao.shape[1:] != (nao, nao):
        raise ValueError("SOC integrals must have shape (3,nao,nao)")

    tol = float(biorth_tol)

    # Biorthonormalize combined occupied space (core + active), preserving core|active ordering.
    mo_occ_bra = np.hstack([mo_core_bra, mo_act_bra])
    mo_occ_ket = np.hstack([mo_core_ket, mo_act_ket])
    nocc = int(mo_occ_bra.shape[1])
    if nocc != ncore + ncas:
        raise RuntimeError("internal error: nocc mismatch")

    o_occ = mo_occ_bra.T @ s_ao @ mo_occ_ket
    l, u = _lu_nopivot(o_occ, tol=tol)
    inv_l = np.linalg.solve(l, np.eye(nocc, dtype=np.float64))
    inv_u = np.linalg.solve(u, np.eye(nocc, dtype=np.float64))
    x_occ_bra = inv_l.T
    x_occ_ket = inv_u

    mo_occ_bra_bi = mo_occ_bra @ x_occ_bra
    mo_occ_ket_bi = mo_occ_ket @ x_occ_ket
    mo_act_bra_bi = mo_occ_bra_bi[:, ncore:]
    mo_act_ket_bi = mo_occ_ket_bi[:, ncore:]

    x_core_bra = x_occ_bra[:ncore, :ncore]
    x_core_ket = x_occ_ket[:ncore, :ncore]
    x_act_bra = x_occ_bra[ncore:, ncore:]
    x_act_ket = x_occ_ket[ncore:, ncore:]

    if ncore:
        det_core_bra = float(np.linalg.det(x_core_bra))
        det_core_ket = float(np.linalg.det(x_core_ket))
        core_scale_bra = det_core_bra ** (-2)
        core_scale_ket = det_core_ket ** (-2)
    else:
        core_scale_bra = 1.0
        core_scale_ket = 1.0

    tra_act_bra = np.linalg.inv(x_act_bra)
    tra_act_ket = np.linalg.inv(x_act_ket)

    ci_bra_bi = transform_csf_ci_for_orbital_transform(bra.drt, bra.ci, tra_act_bra, tol=float(ci_transform_tol))
    ci_ket_bi = transform_csf_ci_for_orbital_transform(ket.drt, ket.ci, tra_act_ket, tol=float(ci_transform_tol))
    ci_bra_bi = np.asarray(ci_bra_bi, dtype=np.float64, order="C") * float(core_scale_bra)
    ci_ket_bi = np.asarray(ci_ket_bi, dtype=np.float64, order="C") * float(core_scale_ket)

    from asuka.soc.trdm import trans_trdm1_triplet_streaming  # noqa: PLC0415

    u_pq = trans_trdm1_triplet_streaming(
        bra.drt,
        ket.drt,
        ci_bra_bi,
        ci_ket_bi,
        block_nops=int(trdm_block_nops),
    )

    h_m_act = np.einsum(
        "ap,mab,bq->mpq",
        np.asarray(mo_act_bra_bi, dtype=np.complex128).conj(),
        h_m_ao,
        np.asarray(mo_act_ket_bi, dtype=np.complex128),
        optimize=True,
    )
    # `trans_trdm1_triplet_*` returns u[p,q] = <bra||T_{q p}||ket> (qp convention), while
    # `h_m_act[p,q]` is built in the standard mixed (bra|h|ket) integral convention. Contract
    # as Σ_{p,q} h[p,q] u[q,p] to match the SI/TRDM qp indexing.
    return np.asarray(np.einsum("mpq,qp->m", h_m_act, u_pq, optimize=True), dtype=np.complex128)


def build_soc_rassi_Gm_biorth_csf(
    states: list[SpinFreeCASSCFStateCSF],
    *,
    s_ao: np.ndarray,
    hso_xyz_ao: np.ndarray | None = None,
    hso_m_ao: np.ndarray | None = None,
    biorth_tol: float = 1e-12,
    ci_transform_tol: float = 1e-14,
    trdm_block_nops: int = 8,
) -> np.ndarray:
    """Build reduced SOC couplings Gm[m,I,J] for a non-orthogonal CSF state basis."""

    if not states:
        raise ValueError("states is empty")

    n = int(len(states))
    gm = np.zeros((n, n, 3), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            gm[i, j] = soc_rassi_gm_pair_biorth_csf(
                states[i],
                states[j],
                s_ao=s_ao,
                hso_xyz_ao=hso_xyz_ao,
                hso_m_ao=hso_m_ao,
                biorth_tol=float(biorth_tol),
                ci_transform_tol=float(ci_transform_tol),
                trdm_block_nops=int(trdm_block_nops),
            )
    return gm


@dataclass(frozen=True)
class _SpinFreeLevel:
    twos: int
    energy: float


def soc_state_interaction_rassi_biorth_csf(
    states: list[SpinFreeCASSCFStateCSF],
    *,
    s_ao: np.ndarray,
    h1_ao: np.ndarray,
    eri_ao: np.ndarray,
    e_nuc: float,
    hso_xyz_ao: np.ndarray | None = None,
    hso_m_ao: np.ndarray | None = None,
    biorth_tol: float = 1e-12,
    ci_transform_tol: float = 1e-14,
    metric_tol: float = 1e-14,
    trdm_block_nops: int = 8,
    symmetrize: bool = True,
) -> SOCRASSIBiorthCSFResult:
    """SOC-SI with a spin-free RASSI orthonormalization step for non-orthogonal CSF states."""

    if not states:
        raise ValueError("states is empty")

    # Spin-free RASSI (H,S) in the non-orthogonal original-state basis.
    h_sf, s_sf = build_spinfree_rassi_matrices_biorth_csf(
        list(states),
        s_ao=s_ao,
        h1_ao=h1_ao,
        eri_ao=eri_ao,
        e_nuc=float(e_nuc),
        biorth_tol=float(biorth_tol),
        ci_transform_tol=float(ci_transform_tol),
    )

    from asuka.soc.si import build_si_hamiltonian_from_Gm, solve_spinfree_state_interaction  # noqa: PLC0415

    # Solve generalized spin-free SI separately per spin block to preserve definite spin labels.
    twos_in = np.asarray([int(st.twos) for st in states], dtype=np.int64)
    unique_twos = sorted(set(int(x) for x in twos_in.tolist()))

    nstate = int(len(states))
    energies_parts: list[np.ndarray] = []
    twos_parts: list[np.ndarray] = []
    mixing_parts: list[tuple[list[int], np.ndarray]] = []

    for tw in unique_twos:
        idxs = [i for i in range(nstate) if int(twos_in[i]) == int(tw)]
        if not idxs:
            continue
        hb = h_sf[np.ix_(idxs, idxs)]
        sb = s_sf[np.ix_(idxs, idxs)]
        evals_b, c_b = solve_spinfree_state_interaction(hb, sb, metric_tol=float(metric_tol))
        energies_parts.append(np.asarray(evals_b, dtype=np.float64))
        twos_parts.append(np.full(int(evals_b.size), int(tw), dtype=np.int64))
        mixing_parts.append((idxs, np.asarray(c_b, dtype=np.float64)))

    if not energies_parts:
        raise RuntimeError("no spin blocks found (unexpected empty spin-free SI)")

    energies = np.concatenate(energies_parts, axis=0)
    twos = np.concatenate(twos_parts, axis=0)

    nroot = int(energies.size)
    mixing = np.zeros((nstate, nroot), dtype=np.float64)
    col = 0
    for idxs, c_b in mixing_parts:
        nb = int(c_b.shape[1])
        mixing[np.asarray(idxs, dtype=np.int64), col : col + nb] = c_b
        col += nb
    if col != nroot:
        raise RuntimeError("internal error assembling spin-free mixing matrix")

    # Sort spin-free RASSI states by energy (matches typical RASSI output order).
    order = np.argsort(energies, kind="stable")
    energies = energies[order]
    twos = twos[order]
    mixing = mixing[:, order]

    sf = SpinFreeRASSIResult(
        energies=np.asarray(energies, dtype=np.float64),
        twos=np.asarray(twos, dtype=np.int64),
        mixing=np.asarray(mixing, dtype=np.float64),
        overlap=np.asarray(s_sf, dtype=np.float64),
        hamiltonian=np.asarray(h_sf, dtype=np.float64),
    )

    # Reduced SOC couplings between original states (non-orthogonal).
    gm_orig = build_soc_rassi_Gm_biorth_csf(
        list(states),
        s_ao=s_ao,
        hso_xyz_ao=hso_xyz_ao,
        hso_m_ao=hso_m_ao,
        biorth_tol=float(biorth_tol),
        ci_transform_tol=float(ci_transform_tol),
        trdm_block_nops=int(trdm_block_nops),
    )

    # Transform reduced couplings to the orthonormalized spin-free RASSI basis.
    gm_rassi = np.einsum("ia,ijm,jb->abm", sf.mixing, gm_orig, sf.mixing, optimize=True)

    sf_levels = [_SpinFreeLevel(twos=int(sf.twos[i]), energy=float(sf.energies[i])) for i in range(nroot)]
    h_si, basis = build_si_hamiltonian_from_Gm(
        sf_levels,
        gm_rassi,  # (nroot,nroot,3)
        include_diag=True,
        symmetrize=bool(symmetrize),
    )
    e_so, c_so = np.linalg.eigh(h_si)

    return SOCRASSIBiorthCSFResult(
        spinfree=sf,
        gm_original=np.asarray(gm_orig, dtype=np.complex128),
        gm_rassi=np.asarray(gm_rassi, dtype=np.complex128),
        so_energies=np.asarray(e_so, dtype=np.complex128),
        so_vectors=np.asarray(c_so, dtype=np.complex128),
        so_basis=list(basis),
    )
