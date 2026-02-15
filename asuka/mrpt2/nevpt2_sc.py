from __future__ import annotations

import numpy as np

from asuka.mrpt2.df_pair_block import DFPairBlock
from asuka.mrpt2.nevpt2_sc_df_tiled import (
    sijrs0_energy_df_tiled,
    srs_m2_energy_df_tiled,
    srsi_m1_energy_df_tiled,
    sijr_p1_energy_df_tiled,
    sir_0_energy_df_tiled,
    sij_p2_energy_df_tiled,
    sr_m1_prime_energy_df_tiled,
    si_p1_prime_energy_df_tiled,
)


_NUMERICAL_ZERO = 1e-14


def _norm_to_energy(norm: np.ndarray, h: np.ndarray, diff: np.ndarray) -> tuple[float, float]:
    """Match PySCF's SC-NEVPT2 Hylleraas reduction: E = -Σ norm/(diff + h/norm)."""

    norm = np.asarray(norm, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    diff = np.asarray(diff, dtype=np.float64)
    if norm.shape != h.shape or norm.shape != diff.shape:
        raise ValueError("norm/h/diff shape mismatch")
    idx = np.abs(norm) > _NUMERICAL_ZERO
    if np.any(idx):
        e2 = -np.sum(norm[idx] / (diff[idx] + h[idx] / norm[idx]))
    else:
        e2 = 0.0
    return float(np.sum(norm)), float(e2)


def make_hdm1(dm1: np.ndarray) -> np.ndarray:
    """Hole 1-RDM-like object used in several SC-NEVPT2 intermediates (PySCF convention)."""

    dm1 = np.asarray(dm1, dtype=np.float64)
    if dm1.ndim != 2 or dm1.shape[0] != dm1.shape[1]:
        raise ValueError("dm1 must be square")
    delta = np.eye(dm1.shape[0], dtype=np.float64)
    return 2.0 * delta - dm1.T


def make_a3(
    h1e: np.ndarray,
    h2e: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
    hdm1: np.ndarray,
) -> np.ndarray:
    """Active-space intermediate A3 (PySCF SC-NEVPT2 convention)."""

    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    hdm1 = np.asarray(hdm1, dtype=np.float64)
    if h1e.shape != dm1.shape or hdm1.shape != dm1.shape:
        raise ValueError("h1e/dm1/hdm1 shape mismatch")
    n = int(dm1.shape[0])
    if h2e.shape != (n, n, n, n) or dm2.shape != (n, n, n, n):
        raise ValueError("h2e/dm2 shape mismatch")
    delta = np.eye(n, dtype=np.float64)
    a3 = (
        np.einsum("ia,ip->pa", h1e, hdm1, optimize=True)
        + 2.0 * np.einsum("ijka,pj,ik->pa", h2e, delta, dm1, optimize=True)
        - np.einsum("ijka,jpik->pa", h2e, dm2, optimize=True)
    )
    return np.asarray(a3, dtype=np.float64, order="C")


def make_hdm2(dm1: np.ndarray, dm2: np.ndarray) -> np.ndarray:
    """Hole 2-RDM-like object (PySCF SC-NEVPT2 convention)."""

    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    if dm1.ndim != 2 or dm1.shape[0] != dm1.shape[1]:
        raise ValueError("dm1 must be square")
    n = int(dm1.shape[0])
    if dm2.shape != (n, n, n, n):
        raise ValueError("dm2 shape mismatch")
    delta = np.eye(n, dtype=np.float64)
    dm2n = np.einsum("ikjl->ijkl", dm2, optimize=True) - np.einsum("jk,il->ijkl", delta, dm1, optimize=True)
    hdm2 = (
        np.einsum("klij->ijkl", dm2n, optimize=True)
        + np.einsum("il,kj->ijkl", delta, dm1, optimize=True)
        + np.einsum("jk,li->ijkl", delta, dm1, optimize=True)
        - 2.0 * np.einsum("ik,lj->ijkl", delta, dm1, optimize=True)
        - 2.0 * np.einsum("jl,ki->ijkl", delta, dm1, optimize=True)
        - 2.0 * np.einsum("il,jk->ijkl", delta, delta, optimize=True)
        + 4.0 * np.einsum("ik,jl->ijkl", delta, delta, optimize=True)
    )
    return np.asarray(hdm2, dtype=np.float64, order="C")


def make_hdm3(
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    hdm1: np.ndarray,
    hdm2: np.ndarray,
) -> np.ndarray:
    """Hole 3-RDM-like object (PySCF SC-NEVPT2 convention)."""

    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    hdm1 = np.asarray(hdm1, dtype=np.float64)
    hdm2 = np.asarray(hdm2, dtype=np.float64)
    n = int(dm1.shape[0])
    if dm1.shape != (n, n) or hdm1.shape != (n, n):
        raise ValueError("dm1/hdm1 shape mismatch")
    if dm2.shape != (n, n, n, n) or hdm2.shape != (n, n, n, n):
        raise ValueError("dm2/hdm2 shape mismatch")
    if dm3.shape != (n, n, n, n, n, n):
        raise ValueError("dm3 shape mismatch")
    delta = np.eye(n, dtype=np.float64)
    hdm3 = (
        -np.einsum("pb,qrac->pqrabc", delta, hdm2, optimize=True)
        - np.einsum("br,pqac->pqrabc", delta, hdm2, optimize=True)
        + 2.0 * np.einsum("bq,prac->pqrabc", delta, hdm2, optimize=True)
        + 2.0 * np.einsum("ap,bqcr->pqrabc", delta, dm2, optimize=True)
        - 4.0 * np.einsum("ap,cr,bq->pqrabc", delta, delta, dm1, optimize=True)
        + 2.0 * np.einsum("cr,bqap->pqrabc", delta, dm2, optimize=True)
        - np.einsum("bqapcr->pqrabc", dm3, optimize=True)
        + 2.0 * np.einsum("ar,pc,bq->pqrabc", delta, delta, dm1, optimize=True)
        - np.einsum("ar,bqcp->pqrabc", delta, dm2, optimize=True)
    )
    return np.asarray(hdm3, dtype=np.float64, order="C")


def make_k27(
    h1e: np.ndarray,
    h2e: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
) -> np.ndarray:
    """Active-space intermediate K27 (PySCF SC-NEVPT2 convention)."""

    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    if h1e.shape != dm1.shape:
        raise ValueError("h1e/dm1 shape mismatch")
    n = int(dm1.shape[0])
    if h2e.shape != (n, n, n, n) or dm2.shape != (n, n, n, n):
        raise ValueError("h2e/dm2 shape mismatch")
    k27 = (
        -np.einsum("ai,pi->pa", h1e, dm1, optimize=True)
        - np.einsum("iajk,pkij->pa", h2e, dm2, optimize=True)
        + np.einsum("iaji,pj->pa", h2e, dm1, optimize=True)
    )
    return np.asarray(k27, dtype=np.float64, order="C")


def make_a7(
    h1e: np.ndarray,
    h2e: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Active-space intermediate A7 and the associated rm2 (PySCF SC-NEVPT2 convention)."""

    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    n = int(dm1.shape[0])
    if h1e.shape != (n, n) or h2e.shape != (n, n, n, n):
        raise ValueError("active integral shape mismatch")
    if dm2.shape != (n, n, n, n) or dm3.shape != (n, n, n, n, n, n):
        raise ValueError("dm2/dm3 shape mismatch")

    # Match PySCF's "norm order" transforms.
    delta = np.eye(n, dtype=np.float64)
    rm2 = np.einsum("iljk->ijkl", dm2, optimize=True) - np.einsum("ik,jl->ijkl", dm1, delta, optimize=True)
    rm3 = (
        np.einsum("injmkl->ijklmn", dm3, optimize=True)
        - np.einsum("jn,imkl->ijklmn", delta, dm2, optimize=True)
        - np.einsum("km,ijln->ijklmn", delta, rm2, optimize=True)
        - np.einsum("kn,ijml->ijklmn", delta, rm2, optimize=True)
    )

    a7 = (
        -np.einsum("bi,pqia->pqab", h1e, rm2, optimize=True)
        - np.einsum("ai,pqbi->pqab", h1e, rm2, optimize=True)
        - np.einsum("kbij,pqkija->pqab", h2e, rm3, optimize=True)
        - np.einsum("kaij,pqkibj->pqab", h2e, rm3, optimize=True)
        - np.einsum("baij,pqij->pqab", h2e, rm2, optimize=True)
    )
    return np.asarray(rm2, dtype=np.float64, order="C"), np.asarray(a7, dtype=np.float64, order="C")


def make_a9(
    h1e: np.ndarray,
    h2e: np.ndarray,
    hdm1: np.ndarray,
    hdm2: np.ndarray,
    hdm3: np.ndarray,
) -> np.ndarray:
    """Active-space intermediate A9 (PySCF SC-NEVPT2 convention)."""

    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    hdm1 = np.asarray(hdm1, dtype=np.float64)
    hdm2 = np.asarray(hdm2, dtype=np.float64)
    hdm3 = np.asarray(hdm3, dtype=np.float64)
    n = int(h1e.shape[0])
    if h1e.shape != (n, n) or hdm1.shape != (n, n):
        raise ValueError("h1e/hdm1 shape mismatch")
    if h2e.shape != (n, n, n, n) or hdm2.shape != (n, n, n, n) or hdm3.shape != (n, n, n, n, n, n):
        raise ValueError("h2e/hdm2/hdm3 shape mismatch")
    a9 = np.einsum("ib,pqai->pqab", h1e, hdm2, optimize=True)
    a9 += 2.0 * np.einsum("ijib,pqaj->pqab", h2e, hdm2, optimize=True)
    a9 -= np.einsum("ijjb,pqai->pqab", h2e, hdm2, optimize=True)
    a9 -= np.einsum("ijkb,pkqaij->pqab", h2e, hdm3, optimize=True)
    a9 += np.einsum("ia,pqib->pqab", h1e, hdm2, optimize=True)
    a9 -= np.einsum("ijja,pqib->pqab", h2e, hdm2, optimize=True)
    a9 -= np.einsum("ijba,pqji->pqab", h2e, hdm2, optimize=True)
    a9 += 2.0 * np.einsum("ijia,pqjb->pqab", h2e, hdm2, optimize=True)
    a9 -= np.einsum("ijka,pqkjbi->pqab", h2e, hdm3, optimize=True)
    return np.asarray(a9, dtype=np.float64, order="C")


def make_a12(
    h1e: np.ndarray,
    h2e: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
) -> np.ndarray:
    """Active-space intermediate A12 (PySCF SC-NEVPT2 convention)."""

    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    n = int(dm1.shape[0])
    if h1e.shape != (n, n) or h2e.shape != (n, n, n, n):
        raise ValueError("active integral shape mismatch")
    if dm2.shape != (n, n, n, n) or dm3.shape != (n, n, n, n, n, n):
        raise ValueError("dm2/dm3 shape mismatch")
    a12 = (
        np.einsum("ia,qpib->pqab", h1e, dm2, optimize=True)
        - np.einsum("bi,qpai->pqab", h1e, dm2, optimize=True)
        + np.einsum("ijka,qpjbik->pqab", h2e, dm3, optimize=True)
        - np.einsum("kbij,qpajki->pqab", h2e, dm3, optimize=True)
        - np.einsum("bjka,qpjk->pqab", h2e, dm2, optimize=True)
        + np.einsum("jbij,qpai->pqab", h2e, dm2, optimize=True)
    )
    return np.asarray(a12, dtype=np.float64, order="C")


def make_a13(
    h1e: np.ndarray,
    h2e: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
) -> np.ndarray:
    """Active-space intermediate A13 (PySCF SC-NEVPT2 convention)."""

    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    n = int(dm1.shape[0])
    if h1e.shape != (n, n) or h2e.shape != (n, n, n, n):
        raise ValueError("active integral shape mismatch")
    if dm2.shape != (n, n, n, n) or dm3.shape != (n, n, n, n, n, n):
        raise ValueError("dm2/dm3 shape mismatch")
    delta = np.eye(n, dtype=np.float64)
    a13 = -np.einsum("ia,qbip->pqab", h1e, dm2, optimize=True)
    a13 += 2.0 * np.einsum("pa,qb->pqab", h1e, dm1, optimize=True)
    a13 += np.einsum("bi,qiap->pqab", h1e, dm2, optimize=True)
    a13 -= 2.0 * np.einsum("pa,bi,qi->pqab", delta, h1e, dm1, optimize=True)
    a13 -= np.einsum("ijka,qbjpik->pqab", h2e, dm3, optimize=True)
    a13 += np.einsum("kbij,qjapki->pqab", h2e, dm3, optimize=True)
    a13 += np.einsum("blma,qmlp->pqab", h2e, dm2, optimize=True)
    a13 += 2.0 * np.einsum("kpma,qbkm->pqab", h2e, dm2, optimize=True)
    a13 -= 2.0 * np.einsum("bpma,qm->pqab", h2e, dm1, optimize=True)
    a13 -= np.einsum("lbkl,qkap->pqab", h2e, dm2, optimize=True)
    a13 -= 2.0 * np.einsum("ap,mbkl,qlmk->pqab", delta, h2e, dm2, optimize=True)
    a13 += 2.0 * np.einsum("ap,lbkl,qk->pqab", delta, h2e, dm1, optimize=True)
    return np.asarray(a13, dtype=np.float64, order="C")


def make_a16(
    h1e: np.ndarray,
    h2e: np.ndarray,
    dm3: np.ndarray,
    *,
    f3ca: np.ndarray,
    f3ac: np.ndarray,
) -> np.ndarray:
    """Active-space intermediate A16 (PySCF SC-NEVPT2 convention).

    Notes
    -----
    This intermediate requires contracted 4-PDM-like objects (`f3ca`/`f3ac`) in
    the same conventions as PySCF's internal `_contract4pdm` helpers.  For
    determinant-FCI validation, these can be obtained via
    `pyscf.mrpt.nevpt2._contract4pdm`.
    """

    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    f3ca = np.asarray(f3ca, dtype=np.float64)
    f3ac = np.asarray(f3ac, dtype=np.float64)

    n = int(h1e.shape[0])
    if h1e.shape != (n, n) or h2e.shape != (n, n, n, n):
        raise ValueError("active integral shape mismatch")
    if dm3.shape != (n, n, n, n, n, n) or f3ca.shape != dm3.shape or f3ac.shape != dm3.shape:
        raise ValueError("dm3/f3ca/f3ac shape mismatch")

    a16 = -np.einsum("ib,rpqiac->pqrabc", h1e, dm3, optimize=True)
    a16 += np.einsum("ia,rpqbic->pqrabc", h1e, dm3, optimize=True)
    a16 -= np.einsum("ci,rpqbai->pqrabc", h1e, dm3, optimize=True)

    # qjkiac = acqjki + delta(ja)qcki + delta(ia)qjkc - delta(qc)ajki - delta(kc)qjai
    # a16 -= einsum('kbij,rpqjkiac->pqrabc', h2e, dm4)  # dm4 path in PySCF
    a16 -= f3ca.transpose(1, 4, 0, 2, 5, 3)  # c'a'acb'b -> a'b'c'abc
    a16 -= np.einsum("kbia,rpqcki->pqrabc", h2e, dm3, optimize=True)
    a16 -= np.einsum("kbaj,rpqjkc->pqrabc", h2e, dm3, optimize=True)
    a16 += np.einsum("cbij,rpqjai->pqrabc", h2e, dm3, optimize=True)
    fdm2 = np.einsum("kbij,rpajki->prab", h2e, dm3, optimize=True)
    for i in range(n):
        a16[:, i, :, :, :, i] += fdm2

    # a16 += einsum('ijka,rpqbjcik->pqrabc', h2e, dm4)
    a16 += f3ac.transpose(1, 2, 0, 4, 3, 5)  # c'a'b'bac -> a'b'c'abc

    # a16 -= einsum('kcij,rpqbajki->pqrabc', h2e, dm4)
    a16 -= f3ca.transpose(1, 2, 0, 4, 3, 5)  # c'a'b'bac -> a'b'c'abc

    a16 += np.einsum("jbij,rpqiac->pqrabc", h2e, dm3, optimize=True)
    a16 -= np.einsum("cjka,rpqbjk->pqrabc", h2e, dm3, optimize=True)
    a16 += np.einsum("jcij,rpqbai->pqrabc", h2e, dm3, optimize=True)
    return np.asarray(a16, dtype=np.float64, order="C")


def make_a17(h1e: np.ndarray, h2e: np.ndarray, dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Active-space intermediate A17 (PySCF SC-NEVPT2 convention)."""

    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    n = int(h1e.shape[0])
    if h1e.shape != (n, n) or h2e.shape != (n, n, n, n):
        raise ValueError("active integral shape mismatch")
    if dm2.shape != (n, n, n, n) or dm3.shape != (n, n, n, n, n, n):
        raise ValueError("dm2/dm3 shape mismatch")

    h1e_eff = h1e - np.einsum("mjjn->mn", h2e, optimize=True)
    a17 = -np.einsum("pi,cabi->abcp", h1e_eff, dm2, optimize=True) - np.einsum(
        "kpij,cabjki->abcp", h2e, dm3, optimize=True
    )
    return np.asarray(a17, dtype=np.float64, order="C")


def make_a19(h1e: np.ndarray, h2e: np.ndarray, dm1: np.ndarray, dm2: np.ndarray) -> np.ndarray:
    """Active-space intermediate A19 (PySCF SC-NEVPT2 convention)."""

    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    n = int(h1e.shape[0])
    if h1e.shape != (n, n) or h2e.shape != (n, n, n, n):
        raise ValueError("active integral shape mismatch")
    if dm1.shape != (n, n) or dm2.shape != (n, n, n, n):
        raise ValueError("dm1/dm2 shape mismatch")

    h1e_eff = h1e - np.einsum("mjjn->mn", h2e, optimize=True)
    a19 = -np.einsum("pi,ai->ap", h1e_eff, dm1, optimize=True) - np.einsum(
        "kpij,ajki->ap", h2e, dm2, optimize=True
    )
    return np.asarray(a19, dtype=np.float64, order="C")


def make_a22(
    h1e: np.ndarray,
    h2e: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    *,
    f3ca: np.ndarray,
    f3ac: np.ndarray,
) -> np.ndarray:
    """Active-space intermediate A22 (PySCF SC-NEVPT2 convention).

    Notes
    -----
    Requires contracted 4-PDM-like objects (`f3ca`/`f3ac`) in PySCF's
    conventions (see `make_a16`).
    """

    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    f3ca = np.asarray(f3ca, dtype=np.float64)
    f3ac = np.asarray(f3ac, dtype=np.float64)

    n = int(h1e.shape[0])
    if h1e.shape != (n, n) or h2e.shape != (n, n, n, n):
        raise ValueError("active integral shape mismatch")
    if dm2.shape != (n, n, n, n) or dm3.shape != (n, n, n, n, n, n):
        raise ValueError("dm2/dm3 shape mismatch")
    if f3ca.shape != dm3.shape or f3ac.shape != dm3.shape:
        raise ValueError("f3ca/f3ac shape mismatch")

    a22 = -np.einsum("pb,kipjac->ijkabc", h1e, dm3, optimize=True)
    a22 -= np.einsum("pa,kibjpc->ijkabc", h1e, dm3, optimize=True)
    a22 += np.einsum("cp,kibjap->ijkabc", h1e, dm3, optimize=True)
    a22 += np.einsum("cqra,kibjqr->ijkabc", h2e, dm3, optimize=True)
    a22 -= np.einsum("qcpq,kibjap->ijkabc", h2e, dm3, optimize=True)

    # qjprac = acqjpr + delta(ja)qcpr + delta(ra)qjpc - delta(qc)ajpr - delta(pc)qjar
    # a22 -= einsum('pqrb,kiqjprac->ijkabc', h2e, dm4)
    a22 -= f3ac.transpose(1, 5, 0, 2, 4, 3)  # c'a'acbb'
    fdm2 = np.einsum("pqrb,kiqcpr->ikbc", h2e, dm3, optimize=True)
    for i in range(n):
        a22[:, i, :, i, :, :] -= fdm2
    a22 -= np.einsum("pqab,kiqjpc->ijkabc", h2e, dm3, optimize=True)
    a22 += np.einsum("pcrb,kiajpr->ijkabc", h2e, dm3, optimize=True)
    a22 += np.einsum("cqrb,kiqjar->ijkabc", h2e, dm3, optimize=True)

    # a22 -= einsum('pqra,kibjqcpr->ijkabc', h2e, dm4)
    a22 -= f3ac.transpose(1, 3, 0, 4, 2, 5)  # c'a'bb'ac -> a'b'c'abc

    # a22 += einsum('rcpq,kibjaqrp->ijkabc', h2e, dm4)
    a22 += f3ca.transpose(1, 3, 0, 4, 2, 5)  # c'a'bb'ac -> a'b'c'abc

    a22 += 2.0 * np.einsum("jb,kiac->ijkabc", h1e, dm2, optimize=True)
    a22 += 2.0 * np.einsum("pjrb,kiprac->ijkabc", h2e, dm3, optimize=True)

    fdm2 = np.einsum("pa,kipc->ikac", h1e, dm2, optimize=True)
    fdm2 -= np.einsum("cp,kiap->ikac", h1e, dm2, optimize=True)
    fdm2 -= np.einsum("cqra,kiqr->ikac", h2e, dm2, optimize=True)
    fdm2 += np.einsum("qcpq,kiap->ikac", h2e, dm2, optimize=True)
    fdm2 += np.einsum("pqra,kiqcpr->ikac", h2e, dm3, optimize=True)
    fdm2 -= np.einsum("rcpq,kiaqrp->ikac", h2e, dm3, optimize=True)
    for i in range(n):
        a22[:, i, :, :, i, :] += fdm2 * 2.0

    return np.asarray(a22, dtype=np.float64, order="C")


def make_a23(h1e: np.ndarray, h2e: np.ndarray, dm1: np.ndarray, dm2: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Active-space intermediate A23 (PySCF SC-NEVPT2 convention)."""

    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    n = int(h1e.shape[0])
    if h1e.shape != (n, n) or h2e.shape != (n, n, n, n):
        raise ValueError("active integral shape mismatch")
    if dm1.shape != (n, n) or dm2.shape != (n, n, n, n) or dm3.shape != (n, n, n, n, n, n):
        raise ValueError("dm shape mismatch")

    a23 = -np.einsum("ip,caib->abcp", h1e, dm2, optimize=True)
    a23 -= np.einsum("pijk,cajbik->abcp", h2e, dm3, optimize=True)
    a23 += 2.0 * np.einsum("bp,ca->abcp", h1e, dm1, optimize=True)
    a23 += 2.0 * np.einsum("pibk,caik->abcp", h2e, dm2, optimize=True)
    return np.asarray(a23, dtype=np.float64, order="C")


def make_a25(h1e: np.ndarray, h2e: np.ndarray, dm1: np.ndarray, dm2: np.ndarray) -> np.ndarray:
    """Active-space intermediate A25 (PySCF SC-NEVPT2 convention)."""

    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    n = int(h1e.shape[0])
    if h1e.shape != (n, n) or h2e.shape != (n, n, n, n):
        raise ValueError("active integral shape mismatch")
    if dm1.shape != (n, n) or dm2.shape != (n, n, n, n):
        raise ValueError("dm shape mismatch")

    a25 = -np.einsum("pi,ai->ap", h1e, dm1, optimize=True)
    a25 -= np.einsum("pijk,jaik->ap", h2e, dm2, optimize=True)
    a25 += 2.0 * np.einsum("ap->pa", h1e, optimize=True)
    a25 += 2.0 * np.einsum("piaj,ij->ap", h2e, dm1, optimize=True)
    return np.asarray(a25, dtype=np.float64, order="C")


def sijrs0_energy_df(
    l_cv: DFPairBlock,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
    *,
    check_shapes: bool = True,
) -> tuple[float, float]:
    """SC-NEVPT2 subspace Sijrs(0): core->virt doubles (MP2-like), DF-backed.

    This implements the same closed-shell contraction pattern used by PySCF's
    SC-NEVPT2 `Sijrs` routine:
      theta = 2*(i a|j b) - (i b|j a)
      t2    = (i a|j b) / (eps_i + eps_j - eps_a - eps_b)
      E     = Σ_{i,j,a,b} t2 * theta
      norm  = Σ_{i,j,a,b} (i a|j b) * theta

    Parameters
    ----------
    l_cv:
        DF pair block for the (core, virt) orbital block, storing d[L, i a].
        Must have shape (ncore*nvirt, naux) with ordered pairs `ia_id = i*nvirt + a`.
    eps_core:
        Semicanonical core orbital energies, shape (ncore,).
    eps_virt:
        Semicanonical virtual orbital energies, shape (nvirt,).
    """
    return sijrs0_energy_df_tiled(l_cv, eps_core, eps_virt)


def _build_h2e_v_sijr_df(l_vc: DFPairBlock, l_ac: DFPairBlock) -> np.ndarray:
    """Build the mixed integral block needed for Sijr(+1) from DF pair blocks.

    Returns
    -------
    h2e_v:
        Array shaped (nvirt, nact, ncore, ncore) with axes interpreted as (r,p,j,i)
        to match the einsum index labels used in PySCF's `Sijr`.
    """

    nvirt, ncore = int(l_vc.nx), int(l_vc.ny)
    nact, ncore2 = int(l_ac.nx), int(l_ac.ny)
    if ncore2 != ncore:
        raise ValueError("l_vc and l_ac core dimensions mismatch")
    if int(l_vc.naux) != int(l_ac.naux):
        raise ValueError("l_vc and l_ac naux mismatch")

    # V[(r,i),(p,j)] = (r i| p j) via DF dot products.
    v = l_vc.l_full @ l_ac.l_full.T  # (nvirt*ncore, nact*ncore)
    v4 = v.reshape(nvirt, ncore, nact, ncore)  # (r,i,p,j)
    # Match PySCF's label convention (r,p,j,i) in the Sijr einsums.
    return np.asarray(v4.transpose(0, 2, 3, 1), order="C")  # (r,p,j,i)


def _build_h2e_v_srsi_df(l_vc: DFPairBlock, l_va: DFPairBlock) -> np.ndarray:
    """Build the mixed integral block needed for Srsi(-1) from DF pair blocks.

    Returns
    -------
    h2e_v:
        Array shaped (nvirt, nvirt, ncore, nact) with axes (r,s,i,p), matching
        the einsum index labels used in PySCF's `Srsi`.
    """

    nvirt, ncore = int(l_vc.nx), int(l_vc.ny)
    nvirt2, nact = int(l_va.nx), int(l_va.ny)
    if nvirt2 != nvirt:
        raise ValueError("l_vc and l_va virt dimensions mismatch")
    if int(l_vc.naux) != int(l_va.naux):
        raise ValueError("l_vc and l_va naux mismatch")

    # V[(r,i),(s,p)] = (r i| s p) via DF dot products.
    v = l_vc.l_full @ l_va.l_full.T  # (nvirt*ncore, nvirt*nact)
    v4 = v.reshape(nvirt, ncore, nvirt, nact)  # (r,i,s,p)
    return np.asarray(v4.transpose(0, 2, 1, 3), order="C")  # (r,s,i,p)


def sijr_p1_energy_from_h2e_v(
    h2e_v: np.ndarray,
    *,
    dm1: np.ndarray,
    dm2: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
) -> tuple[float, float]:
    """SC-NEVPT2 subspace Sijr(+1) from a prebuilt `h2e_v` block (PySCF convention)."""

    h2e_v = np.asarray(h2e_v, dtype=np.float64)
    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    eps_core = np.asarray(eps_core, dtype=np.float64).ravel()
    eps_virt = np.asarray(eps_virt, dtype=np.float64).ravel()

    nvirt, nact, ncore, ncore2 = h2e_v.shape
    if ncore2 != ncore:
        raise ValueError("h2e_v must have a square (core,core) tail")
    if dm1.shape != (nact, nact) or dm2.shape != (nact, nact, nact, nact):
        raise ValueError("dm shape mismatch")
    if h1e.shape != (nact, nact) or h2e.shape != (nact, nact, nact, nact):
        raise ValueError("active integral shape mismatch")
    if eps_core.size != ncore or eps_virt.size != nvirt:
        raise ValueError("orbital energy shape mismatch")

    hdm1 = make_hdm1(dm1)
    a3 = make_a3(h1e, h2e, dm1, dm2, hdm1)

    # Match PySCF's einsums (treat `h2e_v` axes as rpji):
    norm = 2.0 * np.einsum("rpji,raji,pa->rji", h2e_v, h2e_v, hdm1, optimize=True) - 1.0 * np.einsum(
        "rpji,raij,pa->rji", h2e_v, h2e_v, hdm1, optimize=True
    )
    norm = norm + norm.transpose(0, 2, 1)
    ci_diag = np.diag_indices(ncore)
    norm[:, ci_diag[0], ci_diag[1]] *= 0.5

    h = 2.0 * np.einsum("rpji,raji,pa->rji", h2e_v, h2e_v, a3, optimize=True) - 1.0 * np.einsum(
        "rpji,raij,pa->rji", h2e_v, h2e_v, a3, optimize=True
    )
    h = h + h.transpose(0, 2, 1)
    h[:, ci_diag[0], ci_diag[1]] *= 0.5

    diff = eps_virt[:, None, None] - eps_core[None, :, None] - eps_core[None, None, :]
    ci_triu = np.triu_indices(ncore)
    return _norm_to_energy(norm[:, ci_triu[0], ci_triu[1]], h[:, ci_triu[0], ci_triu[1]], diff[:, ci_triu[0], ci_triu[1]])


def srsi_m1_energy_from_h2e_v(
    h2e_v: np.ndarray,
    *,
    dm1: np.ndarray,
    dm2: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
) -> tuple[float, float]:
    """SC-NEVPT2 subspace Srsi(-1) from a prebuilt `h2e_v` block (PySCF convention)."""

    h2e_v = np.asarray(h2e_v, dtype=np.float64)
    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    eps_core = np.asarray(eps_core, dtype=np.float64).ravel()
    eps_virt = np.asarray(eps_virt, dtype=np.float64).ravel()

    nvirt, nvirt2, ncore, nact = h2e_v.shape
    if nvirt2 != nvirt:
        raise ValueError("h2e_v must have a square (virt,virt) head")
    if dm1.shape != (nact, nact) or dm2.shape != (nact, nact, nact, nact):
        raise ValueError("dm shape mismatch")
    if h1e.shape != (nact, nact) or h2e.shape != (nact, nact, nact, nact):
        raise ValueError("active integral shape mismatch")
    if eps_core.size != ncore or eps_virt.size != nvirt:
        raise ValueError("orbital energy shape mismatch")

    k27 = make_k27(h1e, h2e, dm1, dm2)

    norm = 2.0 * np.einsum("rsip,rsia,pa->rsi", h2e_v, h2e_v, dm1, optimize=True) - 1.0 * np.einsum(
        "rsip,sria,pa->rsi", h2e_v, h2e_v, dm1, optimize=True
    )
    norm = norm + norm.transpose(1, 0, 2)
    vi_diag = np.diag_indices(nvirt)
    norm[vi_diag] *= 0.5

    h = 2.0 * np.einsum("rsip,rsia,pa->rsi", h2e_v, h2e_v, k27, optimize=True) - 1.0 * np.einsum(
        "rsip,sria,pa->rsi", h2e_v, h2e_v, k27, optimize=True
    )
    h = h + h.transpose(1, 0, 2)
    h[vi_diag] *= 0.5

    diff = eps_virt[:, None, None] + eps_virt[None, :, None] - eps_core[None, None, :]
    vi_triu = np.triu_indices(nvirt)
    return _norm_to_energy(norm[vi_triu], h[vi_triu], diff[vi_triu])


def sijr_p1_energy_df(
    l_vc: DFPairBlock,
    l_ac: DFPairBlock,
    *,
    dm1: np.ndarray,
    dm2: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
) -> tuple[float, float]:
    """SC-NEVPT2 Sijr(+1) using DF pair blocks for the mixed integrals.

    Uses a tiled DF contraction to avoid materializing the dense
    `(nvirt*ncore, nact*ncore)` mixed-integral block from the naive
    `l_vc.l_full @ l_ac.l_full.T` approach.
    """

    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    nact = int(dm1.shape[0])
    if dm1.shape != (nact, nact) or dm2.shape != (nact, nact, nact, nact):
        raise ValueError("dm shape mismatch")
    if h1e.shape != (nact, nact) or h2e.shape != (nact, nact, nact, nact):
        raise ValueError("active integral shape mismatch")

    hdm1 = make_hdm1(dm1)
    a3 = make_a3(h1e, h2e, dm1, dm2, hdm1)
    return sijr_p1_energy_df_tiled(
        l_vc,
        l_ac,
        hdm1=hdm1,
        a3=a3,
        eps_core=eps_core,
        eps_virt=eps_virt,
        numerical_zero=_NUMERICAL_ZERO,
    )


def srsi_m1_energy_df(
    l_vc: DFPairBlock,
    l_va: DFPairBlock,
    *,
    dm1: np.ndarray,
    dm2: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
) -> tuple[float, float]:
    """SC-NEVPT2 Srsi(-1) using DF pair blocks for the mixed integrals.

    Uses a tiled DF contraction to avoid materializing the dense
    `(nvirt*ncore, nvirt*nact)` mixed-integral block from the naive
    `l_vc.l_full @ l_va.l_full.T` approach.
    """

    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    nact = int(dm1.shape[0])
    if dm1.shape != (nact, nact) or dm2.shape != (nact, nact, nact, nact):
        raise ValueError("dm shape mismatch")
    if h1e.shape != (nact, nact) or h2e.shape != (nact, nact, nact, nact):
        raise ValueError("active integral shape mismatch")

    k27 = make_k27(h1e, h2e, dm1, dm2)
    return srsi_m1_energy_df_tiled(
        l_vc,
        l_va,
        dm1=dm1,
        k27=k27,
        eps_core=eps_core,
        eps_virt=eps_virt,
        numerical_zero=_NUMERICAL_ZERO,
    )


def _build_h2e_v_srs_df(l_va: DFPairBlock) -> np.ndarray:
    """Build the mixed integral block needed for Srs(-2) from DF pair blocks.

    Returns
    -------
    h2e_v:
        Array shaped (nvirt, nvirt, nact, nact) with axes interpreted as (r,s,q,p)
        to match the einsum index labels used in PySCF's `Srs`.
    """

    nvirt, nact = int(l_va.nx), int(l_va.ny)
    # V[(r,p),(s,q)] = (r p| s q) via DF dot products (row/col order is (virt,act)).
    v = l_va.l_full @ l_va.l_full.T  # (nvirt*nact, nvirt*nact)
    v4 = v.reshape(nvirt, nact, nvirt, nact)  # (r,p,s,q)
    # Return (r,s,q,p) so that 'rsqp' and 'rsba' labels match PySCF usage.
    return np.asarray(v4.transpose(0, 2, 3, 1), order="C")


def _build_h2e_v_sij_df(l_ac: DFPairBlock) -> np.ndarray:
    """Build the mixed integral block needed for Sij(+2) from DF pair blocks.

    Returns
    -------
    h2e_v:
        Array shaped (nact, nact, ncore, ncore) with axes (q,p,i,j), matching
        the einsum index labels used in PySCF's `Sij`.
    """

    nact, ncore = int(l_ac.nx), int(l_ac.ny)
    v = l_ac.l_full @ l_ac.l_full.T  # (nact*ncore, nact*ncore)
    v4 = v.reshape(nact, ncore, nact, ncore)  # (p,i,q,j)
    return np.asarray(v4.transpose(2, 0, 1, 3), order="C")  # (q,p,i,j)


def srs_m2_energy_from_h2e_v(
    h2e_v: np.ndarray,
    *,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    eps_virt: np.ndarray,
) -> tuple[float, float]:
    """SC-NEVPT2 subspace Srs(-2) from a prebuilt `h2e_v` block (PySCF convention)."""

    h2e_v = np.asarray(h2e_v, dtype=np.float64)
    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    eps_virt = np.asarray(eps_virt, dtype=np.float64).ravel()

    nvirt, nvirt2, nact, nact2 = h2e_v.shape
    if nvirt2 != nvirt:
        raise ValueError("h2e_v must have a square (virt,virt) head")
    if nact2 != nact:
        raise ValueError("h2e_v must have a square (act,act) tail")
    if eps_virt.size != nvirt:
        raise ValueError("eps_virt shape mismatch")
    if dm1.shape != (nact, nact) or dm2.shape != (nact, nact, nact, nact) or dm3.shape != (nact, nact, nact, nact, nact, nact):
        raise ValueError("dm shape mismatch")
    if h1e.shape != (nact, nact) or h2e.shape != (nact, nact, nact, nact):
        raise ValueError("active integral shape mismatch")

    rm2, a7 = make_a7(h1e, h2e, dm1, dm2, dm3)
    norm = 0.5 * np.einsum("rsqp,rsba,pqba->rs", h2e_v, h2e_v, rm2, optimize=True)
    h = 0.5 * np.einsum("rsqp,rsba,pqab->rs", h2e_v, h2e_v, a7, optimize=True)
    diff = eps_virt[:, None] + eps_virt[None, :]
    return _norm_to_energy(norm, h, diff)


def sij_p2_energy_from_h2e_v(
    h2e_v: np.ndarray,
    *,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    eps_core: np.ndarray,
) -> tuple[float, float]:
    """SC-NEVPT2 subspace Sij(+2) from a prebuilt `h2e_v` block (PySCF convention)."""

    h2e_v = np.asarray(h2e_v, dtype=np.float64)
    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    eps_core = np.asarray(eps_core, dtype=np.float64).ravel()

    nact0, nact1, ncore, ncore2 = h2e_v.shape
    if nact1 != nact0:
        raise ValueError("h2e_v must have a square (act,act) head")
    if ncore2 != ncore:
        raise ValueError("h2e_v must have a square (core,core) tail")
    if eps_core.size != ncore:
        raise ValueError("eps_core shape mismatch")
    nact = int(nact0)
    if dm1.shape != (nact, nact) or dm2.shape != (nact, nact, nact, nact) or dm3.shape != (nact, nact, nact, nact, nact, nact):
        raise ValueError("dm shape mismatch")
    if h1e.shape != (nact, nact) or h2e.shape != (nact, nact, nact, nact):
        raise ValueError("active integral shape mismatch")

    hdm1 = make_hdm1(dm1)
    hdm2 = make_hdm2(dm1, dm2)
    hdm3 = make_hdm3(dm1, dm2, dm3, hdm1, hdm2)
    a9 = make_a9(h1e, h2e, hdm1, hdm2, hdm3)

    norm = 0.5 * np.einsum("qpij,baij,pqab->ij", h2e_v, h2e_v, hdm2, optimize=True)
    h = 0.5 * np.einsum("qpij,baij,pqab->ij", h2e_v, h2e_v, a9, optimize=True)
    diff = eps_core[:, None] + eps_core[None, :]
    return _norm_to_energy(norm, h, -diff)


def srs_m2_energy_df(
    l_va: DFPairBlock,
    *,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    eps_virt: np.ndarray,
) -> tuple[float, float]:
    """SC-NEVPT2 Srs(-2) using DF pair blocks for the mixed integrals.

    Uses a tiled DF contraction to avoid materializing the dense
    `(nvirt*nact, nvirt*nact)` block required by the naive
    `l_va.l_full @ l_va.l_full.T` approach.
    """

    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    nact = int(dm1.shape[0])
    if dm1.shape != (nact, nact) or dm2.shape != (nact, nact, nact, nact) or dm3.shape != (nact, nact, nact, nact, nact, nact):
        raise ValueError("dm shape mismatch")
    if h1e.shape != (nact, nact) or h2e.shape != (nact, nact, nact, nact):
        raise ValueError("active integral shape mismatch")

    rm2, a7 = make_a7(h1e, h2e, dm1, dm2, dm3)
    n2 = nact * nact
    m_norm = np.asarray(rm2.transpose(0, 1, 3, 2).reshape(n2, n2), order="C")
    m_h = np.asarray(a7.reshape(n2, n2), order="C")
    return srs_m2_energy_df_tiled(l_va, m_norm=m_norm, m_h=m_h, eps_virt=eps_virt, numerical_zero=_NUMERICAL_ZERO)


def sij_p2_energy_df(
    l_ac: DFPairBlock,
    *,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    eps_core: np.ndarray,
) -> tuple[float, float]:
    """SC-NEVPT2 Sij(+2) using DF pair blocks for the mixed integrals.

    Uses a tiled DF contraction to avoid materializing the dense
    `(nact*ncore, nact*ncore)` intermediate from the naive `l_ac @ l_ac^T`
    approach.
    """

    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    nact = int(dm1.shape[0])
    if dm1.shape != (nact, nact) or dm2.shape != (nact, nact, nact, nact) or dm3.shape != (nact, nact, nact, nact, nact, nact):
        raise ValueError("dm shape mismatch")
    if h1e.shape != (nact, nact) or h2e.shape != (nact, nact, nact, nact):
        raise ValueError("active integral shape mismatch")

    hdm1 = make_hdm1(dm1)
    hdm2 = make_hdm2(dm1, dm2)
    hdm3 = make_hdm3(dm1, dm2, dm3, hdm1, hdm2)
    a9 = make_a9(h1e, h2e, hdm1, hdm2, hdm3)

    n2 = nact * nact
    m_norm = np.asarray(hdm2.reshape(n2, n2), order="C")
    m_h = np.asarray(a9.reshape(n2, n2), order="C")
    return sij_p2_energy_df_tiled(l_ac, m_norm=m_norm, m_h=m_h, eps_core=eps_core, numerical_zero=_NUMERICAL_ZERO)


def _build_h2e_v1_sir_df(l_vc: DFPairBlock, l_aa: DFPairBlock) -> np.ndarray:
    """Build the `(virt,core | act,act)` block for Sir(0).

    Returns
    -------
    h2e_v1:
        Shape (nvirt, nact, ncore, nact) with axes (r,p,i,q), matching the
        einsum index labels used in PySCF's `Sir` (`rpiq`).
    """

    nvirt, ncore = int(l_vc.nx), int(l_vc.ny)
    nact0, nact1 = int(l_aa.nx), int(l_aa.ny)
    if nact0 != nact1:
        raise ValueError("l_aa must be square (act,act)")
    if int(l_vc.naux) != int(l_aa.naux):
        raise ValueError("l_vc and l_aa naux mismatch")
    nact = nact0

    v = l_vc.l_full @ l_aa.l_full.T  # (nvirt*ncore, nact*nact)
    v4 = v.reshape(nvirt, ncore, nact, nact)  # (r,i,p,q)
    return np.asarray(v4.transpose(0, 2, 1, 3), order="C")  # (r,p,i,q)


def _build_h2e_v2_sir_df(l_va: DFPairBlock, l_ac: DFPairBlock) -> np.ndarray:
    """Build the `(virt,act | act,core)` block for Sir(0).

    Returns
    -------
    h2e_v2:
        Shape (nvirt, nact, nact, ncore) with axes (r,p,q,i), matching the
        einsum index labels used in PySCF's `Sir` (`rpqi`).
    """

    nvirt, nact = int(l_va.nx), int(l_va.ny)
    nact2, ncore = int(l_ac.nx), int(l_ac.ny)
    if nact2 != nact:
        raise ValueError("l_va and l_ac active dimensions mismatch")
    if int(l_va.naux) != int(l_ac.naux):
        raise ValueError("l_va and l_ac naux mismatch")

    v = l_va.l_full @ l_ac.l_full.T  # (nvirt*nact, nact*ncore)
    return np.asarray(v.reshape(nvirt, nact, nact, ncore), order="C")  # (r,p,q,i)


def sir_0_energy_from_integrals(
    h2e_v1: np.ndarray,
    h2e_v2: np.ndarray,
    h1e_v: np.ndarray,
    *,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
) -> tuple[float, float]:
    """SC-NEVPT2 subspace Sir(0) from prebuilt mixed-integral blocks (PySCF convention)."""

    h2e_v1 = np.asarray(h2e_v1, dtype=np.float64)
    h2e_v2 = np.asarray(h2e_v2, dtype=np.float64)
    h1e_v = np.asarray(h1e_v, dtype=np.float64)
    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    eps_core = np.asarray(eps_core, dtype=np.float64).ravel()
    eps_virt = np.asarray(eps_virt, dtype=np.float64).ravel()

    nvirt, nact, ncore, nact2 = h2e_v1.shape
    if nact2 != nact:
        raise ValueError("h2e_v1 last axis must match nact")
    nvirt2, nact3, nact4, ncore2 = h2e_v2.shape
    if nvirt2 != nvirt or nact3 != nact or nact4 != nact or ncore2 != ncore:
        raise ValueError("h2e_v2 shape mismatch")
    if h1e_v.shape != (nvirt, ncore):
        raise ValueError("h1e_v shape mismatch (expect virt x core)")
    if eps_core.size != ncore or eps_virt.size != nvirt:
        raise ValueError("orbital energy shape mismatch")
    if dm1.shape != (nact, nact) or dm2.shape != (nact, nact, nact, nact) or dm3.shape != (nact, nact, nact, nact, nact, nact):
        raise ValueError("dm shape mismatch")
    if h1e.shape != (nact, nact) or h2e.shape != (nact, nact, nact, nact):
        raise ValueError("active integral shape mismatch")

    norm = (
        2.0 * np.einsum("rpiq,raib,qpab->ir", h2e_v1, h2e_v1, dm2, optimize=True)
        - np.einsum("rpiq,rabi,qpab->ir", h2e_v1, h2e_v2, dm2, optimize=True)
        - np.einsum("rpqi,raib,qpab->ir", h2e_v2, h2e_v1, dm2, optimize=True)
        + 2.0 * np.einsum("raqi,rabi,qb->ir", h2e_v2, h2e_v2, dm1, optimize=True)
        - np.einsum("rpqi,rabi,qbap->ir", h2e_v2, h2e_v2, dm2, optimize=True)
        + np.einsum("rpqi,raai,qp->ir", h2e_v2, h2e_v2, dm1, optimize=True)
        + 4.0 * np.einsum("rpiq,ri,qp->ir", h2e_v1, h1e_v, dm1, optimize=True)
        - 2.0 * np.einsum("rpqi,ri,qp->ir", h2e_v2, h1e_v, dm1, optimize=True)
        + 2.0 * np.einsum("ri,ri->ir", h1e_v, h1e_v, optimize=True)
    )

    a12 = make_a12(h1e, h2e, dm1, dm2, dm3)
    a13 = make_a13(h1e, h2e, dm1, dm2, dm3)

    h = (
        2.0 * np.einsum("rpiq,raib,pqab->ir", h2e_v1, h2e_v1, a12, optimize=True)
        - np.einsum("rpiq,rabi,pqab->ir", h2e_v1, h2e_v2, a12, optimize=True)
        - np.einsum("rpqi,raib,pqab->ir", h2e_v2, h2e_v1, a12, optimize=True)
        + np.einsum("rpqi,rabi,pqab->ir", h2e_v2, h2e_v2, a13, optimize=True)
    )

    diff = eps_core[:, None] - eps_virt[None, :]
    return _norm_to_energy(norm, h, -diff)


def sir_0_energy_df(
    l_vc: DFPairBlock,
    l_aa: DFPairBlock,
    l_va: DFPairBlock,
    l_ac: DFPairBlock,
    h1e_v: np.ndarray,
    *,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
) -> tuple[float, float]:
    """SC-NEVPT2 Sir(0) using DF pair blocks for the mixed integrals.

    Uses a tiled DF contraction to avoid materializing the dense mixed-integral
    tensors required by the naive `l_vc @ l_aa^T` / `l_va @ l_ac^T` approach.
    """

    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    nact = int(dm1.shape[0])
    if dm1.shape != (nact, nact) or dm2.shape != (nact, nact, nact, nact) or dm3.shape != (nact, nact, nact, nact, nact, nact):
        raise ValueError("dm shape mismatch")
    if h1e.shape != (nact, nact) or h2e.shape != (nact, nact, nact, nact):
        raise ValueError("active integral shape mismatch")

    a12 = make_a12(h1e, h2e, dm1, dm2, dm3)
    a13 = make_a13(h1e, h2e, dm1, dm2, dm3)
    return sir_0_energy_df_tiled(
        l_vc,
        l_aa,
        l_va,
        l_ac,
        h1e_v,
        dm1=dm1,
        dm2=dm2,
        a12=a12,
        a13=a13,
        eps_core=eps_core,
        eps_virt=eps_virt,
        numerical_zero=_NUMERICAL_ZERO,
    )


def _build_h2e_v_sr_df(l_va: DFPairBlock, l_aa: DFPairBlock) -> np.ndarray:
    """Build the `(virt,act | act,act)` block for Sr(-1)'.

    Returns
    -------
    h2e_v:
        Shape (nvirt, nact, nact, nact) with axes (r,p,q,s), matching the
        einsum index labels used in PySCF's `Sr` (`rpqs`, with the first index
        being the external virtual label).
    """

    nvirt, nact = int(l_va.nx), int(l_va.ny)
    nact0, nact1 = int(l_aa.nx), int(l_aa.ny)
    if nact0 != nact1 or nact0 != nact:
        raise ValueError("l_va/l_aa active dimensions mismatch")
    if int(l_va.naux) != int(l_aa.naux):
        raise ValueError("l_va and l_aa naux mismatch")

    v = l_va.l_full @ l_aa.l_full.T  # (nvirt*nact, nact*nact)
    v4 = v.reshape(nvirt, nact, nact, nact)  # (r,q,p,s)
    return np.asarray(v4.transpose(0, 2, 1, 3), order="C")  # (r,p,q,s)


def sr_m1_prime_energy_from_integrals(
    h2e_v: np.ndarray,
    h1e_v: np.ndarray,
    *,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    eps_virt: np.ndarray,
    f3ca: np.ndarray,
    f3ac: np.ndarray,
) -> tuple[float, float]:
    """SC-NEVPT2 subspace Sr(-1)' from prebuilt mixed-integral blocks (PySCF convention)."""

    h2e_v = np.asarray(h2e_v, dtype=np.float64)
    h1e_v = np.asarray(h1e_v, dtype=np.float64)
    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    eps_virt = np.asarray(eps_virt, dtype=np.float64).ravel()
    f3ca = np.asarray(f3ca, dtype=np.float64)
    f3ac = np.asarray(f3ac, dtype=np.float64)

    nvirt, nact, nact2, nact3 = h2e_v.shape
    if nact2 != nact or nact3 != nact:
        raise ValueError("h2e_v active axes mismatch")
    if h1e_v.shape != (nvirt, nact):
        raise ValueError("h1e_v shape mismatch (expect virt x act)")
    if eps_virt.size != nvirt:
        raise ValueError("eps_virt shape mismatch")
    if dm1.shape != (nact, nact) or dm2.shape != (nact, nact, nact, nact) or dm3.shape != (nact, nact, nact, nact, nact, nact):
        raise ValueError("dm shape mismatch")
    if h1e.shape != (nact, nact) or h2e.shape != (nact, nact, nact, nact):
        raise ValueError("active integral shape mismatch")
    if f3ca.shape != dm3.shape or f3ac.shape != dm3.shape:
        raise ValueError("f3ca/f3ac shape mismatch")

    a16 = make_a16(h1e, h2e, dm3, f3ca=f3ca, f3ac=f3ac)
    a17 = make_a17(h1e, h2e, dm2, dm3)
    a19 = make_a19(h1e, h2e, dm1, dm2)

    h = (
        np.einsum("ipqr,pqrabc,iabc->i", h2e_v, a16, h2e_v, optimize=True)
        + 2.0 * np.einsum("ipqr,pqra,ia->i", h2e_v, a17, h1e_v, optimize=True)
        + np.einsum("ip,pa,ia->i", h1e_v, a19, h1e_v, optimize=True)
    )

    norm = (
        np.einsum("ipqr,rpqbac,iabc->i", h2e_v, dm3, h2e_v, optimize=True)
        + 2.0 * np.einsum("ipqr,rpqa,ia->i", h2e_v, dm2, h1e_v, optimize=True)
        + np.einsum("ip,pa,ia->i", h1e_v, dm1, h1e_v, optimize=True)
    )

    return _norm_to_energy(norm, h, eps_virt)


def sr_m1_prime_energy_df(
    l_va: DFPairBlock,
    l_aa: DFPairBlock,
    h1e_v: np.ndarray,
    *,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    eps_virt: np.ndarray,
    f3ca: np.ndarray,
    f3ac: np.ndarray,
) -> tuple[float, float]:
    """SC-NEVPT2 Sr(-1)' using DF pair blocks for the mixed integrals.

    Uses a tiled DF contraction to avoid materializing the dense
    `(nvirt*nact, nact*nact)` intermediate from the naive `l_va @ l_aa^T` build.
    """

    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    f3ca = np.asarray(f3ca, dtype=np.float64)
    f3ac = np.asarray(f3ac, dtype=np.float64)
    nact = int(dm1.shape[0])
    if dm1.shape != (nact, nact) or dm2.shape != (nact, nact, nact, nact) or dm3.shape != (nact, nact, nact, nact, nact, nact):
        raise ValueError("dm shape mismatch")
    if h1e.shape != (nact, nact) or h2e.shape != (nact, nact, nact, nact):
        raise ValueError("active integral shape mismatch")
    if f3ca.shape != dm3.shape or f3ac.shape != dm3.shape:
        raise ValueError("f3ca/f3ac shape mismatch")

    a16 = make_a16(h1e, h2e, dm3, f3ca=f3ca, f3ac=f3ac)
    a17 = make_a17(h1e, h2e, dm2, dm3)
    a19 = make_a19(h1e, h2e, dm1, dm2)
    return sr_m1_prime_energy_df_tiled(
        l_va,
        l_aa,
        h1e_v,
        dm1=dm1,
        dm2=dm2,
        dm3=dm3,
        a16=a16,
        a17=a17,
        a19=a19,
        eps_virt=eps_virt,
        numerical_zero=_NUMERICAL_ZERO,
    )


def _build_h2e_v_si_df(l_ac: DFPairBlock, l_aa: DFPairBlock) -> np.ndarray:
    """Build the `(act,core | act,act)` block for Si(+1)'.

    Returns
    -------
    h2e_v:
        Shape (nact, nact, ncore, nact) with axes (q,p,i,r), matching the
        einsum index labels used in PySCF's `Si` (`qpir`).
    """

    nact, ncore = int(l_ac.nx), int(l_ac.ny)
    nact0, nact1 = int(l_aa.nx), int(l_aa.ny)
    if nact0 != nact1 or nact0 != nact:
        raise ValueError("l_ac/l_aa active dimensions mismatch")
    if int(l_ac.naux) != int(l_aa.naux):
        raise ValueError("l_ac and l_aa naux mismatch")

    v = l_ac.l_full @ l_aa.l_full.T  # (nact*ncore, nact*nact)
    v4 = v.reshape(nact, ncore, nact, nact)  # (q,i,p,r)
    return np.asarray(v4.transpose(0, 2, 1, 3), order="C")  # (q,p,i,r)


def si_p1_prime_energy_from_integrals(
    h2e_v: np.ndarray,
    h1e_v: np.ndarray,
    *,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    eps_core: np.ndarray,
    f3ca: np.ndarray,
    f3ac: np.ndarray,
) -> tuple[float, float]:
    """SC-NEVPT2 subspace Si(+1)' from prebuilt mixed-integral blocks (PySCF convention)."""

    h2e_v = np.asarray(h2e_v, dtype=np.float64)
    h1e_v = np.asarray(h1e_v, dtype=np.float64)
    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    eps_core = np.asarray(eps_core, dtype=np.float64).ravel()
    f3ca = np.asarray(f3ca, dtype=np.float64)
    f3ac = np.asarray(f3ac, dtype=np.float64)

    nact0, nact1, ncore, nact2 = h2e_v.shape
    if nact0 != nact1 or nact0 != nact2:
        raise ValueError("h2e_v active axes mismatch")
    nact = nact0
    if h1e_v.shape != (nact, ncore):
        raise ValueError("h1e_v shape mismatch (expect act x core)")
    if eps_core.size != ncore:
        raise ValueError("eps_core shape mismatch")
    if dm1.shape != (nact, nact) or dm2.shape != (nact, nact, nact, nact) or dm3.shape != (nact, nact, nact, nact, nact, nact):
        raise ValueError("dm shape mismatch")
    if h1e.shape != (nact, nact) or h2e.shape != (nact, nact, nact, nact):
        raise ValueError("active integral shape mismatch")
    if f3ca.shape != dm3.shape or f3ac.shape != dm3.shape:
        raise ValueError("f3ca/f3ac shape mismatch")

    a22 = make_a22(h1e, h2e, dm2, dm3, f3ca=f3ca, f3ac=f3ac)
    a23 = make_a23(h1e, h2e, dm1, dm2, dm3)
    a25 = make_a25(h1e, h2e, dm1, dm2)

    delta = np.eye(nact, dtype=np.float64)
    dm3_h = 2.0 * np.einsum("abef,cd->abcdef", dm2, delta, optimize=True) - dm3.transpose(0, 1, 3, 2, 4, 5)
    dm2_h = 2.0 * np.einsum("ab,cd->abcd", dm1, delta, optimize=True) - dm2.transpose(0, 1, 3, 2)
    dm1_h = 2.0 * delta - dm1.T

    h = (
        np.einsum("qpir,pqrabc,baic->i", h2e_v, a22, h2e_v, optimize=True)
        + 2.0 * np.einsum("qpir,pqra,ai->i", h2e_v, a23, h1e_v, optimize=True)
        + np.einsum("pi,pa,ai->i", h1e_v, a25, h1e_v, optimize=True)
    )

    norm = (
        np.einsum("qpir,rpqbac,baic->i", h2e_v, dm3_h, h2e_v, optimize=True)
        + 2.0 * np.einsum("qpir,rpqa,ai->i", h2e_v, dm2_h, h1e_v, optimize=True)
        + np.einsum("pi,pa,ai->i", h1e_v, dm1_h, h1e_v, optimize=True)
    )

    return _norm_to_energy(norm, h, -eps_core)


def si_p1_prime_energy_df(
    l_ac: DFPairBlock,
    l_aa: DFPairBlock,
    h1e_v: np.ndarray,
    *,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    eps_core: np.ndarray,
    f3ca: np.ndarray,
    f3ac: np.ndarray,
) -> tuple[float, float]:
    """SC-NEVPT2 Si(+1)' using DF pair blocks for the mixed integrals.

    Uses a tiled DF contraction to avoid materializing the dense
    `(nact*ncore, nact*nact)` intermediate from the naive `l_ac @ l_aa^T` build.
    """

    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    f3ca = np.asarray(f3ca, dtype=np.float64)
    f3ac = np.asarray(f3ac, dtype=np.float64)
    nact = int(dm1.shape[0])
    if dm1.shape != (nact, nact) or dm2.shape != (nact, nact, nact, nact) or dm3.shape != (nact, nact, nact, nact, nact, nact):
        raise ValueError("dm shape mismatch")
    if h1e.shape != (nact, nact) or h2e.shape != (nact, nact, nact, nact):
        raise ValueError("active integral shape mismatch")
    if f3ca.shape != dm3.shape or f3ac.shape != dm3.shape:
        raise ValueError("f3ca/f3ac shape mismatch")

    a22 = make_a22(h1e, h2e, dm2, dm3, f3ca=f3ca, f3ac=f3ac)
    a23 = make_a23(h1e, h2e, dm1, dm2, dm3)
    a25 = make_a25(h1e, h2e, dm1, dm2)

    delta = np.eye(nact, dtype=np.float64)
    dm3_h = 2.0 * np.einsum("abef,cd->abcdef", dm2, delta, optimize=True) - dm3.transpose(0, 1, 3, 2, 4, 5)
    dm2_h = 2.0 * np.einsum("ab,cd->abcd", dm1, delta, optimize=True) - dm2.transpose(0, 1, 3, 2)
    dm1_h = 2.0 * delta - dm1.T

    return si_p1_prime_energy_df_tiled(
        l_ac,
        l_aa,
        h1e_v,
        dm1_h=dm1_h,
        dm2_h=dm2_h,
        dm3_h=dm3_h,
        a22=a22,
        a23=a23,
        a25=a25,
        eps_core=eps_core,
        numerical_zero=_NUMERICAL_ZERO,
    )


def nevpt2_sc_total_energy_df(
    *,
    l_cv: DFPairBlock,
    l_vc: DFPairBlock,
    l_va: DFPairBlock,
    l_ac: DFPairBlock,
    l_aa: DFPairBlock,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
    h1e_v_sir: np.ndarray,
    h1e_v_sr: np.ndarray,
    h1e_v_si: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    f3ca: np.ndarray,
    f3ac: np.ndarray,
) -> dict[str, float]:
    """Compute total SC-NEVPT2 (all 8 subspaces) from DF pair blocks and active tensors.

    This is a thin orchestrator around the individual subspace kernels in this
    module.  All arrays are expected to be in the same conventions as used by
    the `*_energy_df` routines (matching PySCF's SC-NEVPT2 implementation).
    """

    norm_sijrs0, e_sijrs0 = sijrs0_energy_df(l_cv, eps_core, eps_virt)
    norm_sijr, e_sijr = sijr_p1_energy_df(
        l_vc, l_ac, dm1=dm1, dm2=dm2, h1e=h1e, h2e=h2e, eps_core=eps_core, eps_virt=eps_virt
    )
    norm_srsi, e_srsi = srsi_m1_energy_df(
        l_vc, l_va, dm1=dm1, dm2=dm2, h1e=h1e, h2e=h2e, eps_core=eps_core, eps_virt=eps_virt
    )
    norm_srs, e_srs = srs_m2_energy_df(l_va, dm1=dm1, dm2=dm2, dm3=dm3, h1e=h1e, h2e=h2e, eps_virt=eps_virt)
    norm_sij, e_sij = sij_p2_energy_df(l_ac, dm1=dm1, dm2=dm2, dm3=dm3, h1e=h1e, h2e=h2e, eps_core=eps_core)
    norm_sir, e_sir = sir_0_energy_df(
        l_vc,
        l_aa,
        l_va,
        l_ac,
        h1e_v_sir,
        dm1=dm1,
        dm2=dm2,
        dm3=dm3,
        h1e=h1e,
        h2e=h2e,
        eps_core=eps_core,
        eps_virt=eps_virt,
    )
    norm_sr, e_sr = sr_m1_prime_energy_df(
        l_va,
        l_aa,
        h1e_v_sr,
        dm1=dm1,
        dm2=dm2,
        dm3=dm3,
        h1e=h1e,
        h2e=h2e,
        eps_virt=eps_virt,
        f3ca=f3ca,
        f3ac=f3ac,
    )
    norm_si, e_si = si_p1_prime_energy_df(
        l_ac,
        l_aa,
        h1e_v_si,
        dm1=dm1,
        dm2=dm2,
        dm3=dm3,
        h1e=h1e,
        h2e=h2e,
        eps_core=eps_core,
        f3ca=f3ca,
        f3ac=f3ac,
    )

    e_total = float(e_sijrs0 + e_sijr + e_srsi + e_srs + e_sij + e_sir + e_sr + e_si)
    norm_total = float(norm_sijrs0 + norm_sijr + norm_srsi + norm_srs + norm_sij + norm_sir + norm_sr + norm_si)
    return {
        "norm_sijrs0": float(norm_sijrs0),
        "e_sijrs0": float(e_sijrs0),
        "norm_sijr_p1": float(norm_sijr),
        "e_sijr_p1": float(e_sijr),
        "norm_srsi_m1": float(norm_srsi),
        "e_srsi_m1": float(e_srsi),
        "norm_srs_m2": float(norm_srs),
        "e_srs_m2": float(e_srs),
        "norm_sij_p2": float(norm_sij),
        "e_sij_p2": float(e_sij),
        "norm_sir_0": float(norm_sir),
        "e_sir_0": float(e_sir),
        "norm_sr_m1_prime": float(norm_sr),
        "e_sr_m1_prime": float(e_sr),
        "norm_si_p1_prime": float(norm_si),
        "e_si_p1_prime": float(e_si),
        "norm_total": norm_total,
        "e_total": e_total,
    }
