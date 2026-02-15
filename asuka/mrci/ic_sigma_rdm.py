from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.cuguga.eri import restore_eri1
from asuka.mrci.generalized_davidson import GeneralizedDavidsonResult, generalized_davidson1
from asuka.mrci.ic_basis import ICSingles, OrbitalSpaces
from asuka.mrci.ic_overlap import apply_overlap_ref_singles


def _as_square_f64(a: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2D array")
    return arr


def _as_dm2_f64(a: Any) -> np.ndarray:
    dm2 = np.asarray(a, dtype=np.float64)
    if dm2.ndim != 4 or dm2.shape[0] != dm2.shape[1] or dm2.shape[0] != dm2.shape[2] or dm2.shape[0] != dm2.shape[3]:
        raise ValueError("dm2 must have shape (nI, nI, nI, nI)")
    return dm2


def _compute_m_singles_diag(
    *,
    h1e: np.ndarray,
    eri4: np.ndarray,
    e_ref: float,
    gamma: np.ndarray,
    dm2: np.ndarray,
    spaces: OrbitalSpaces,
) -> np.ndarray:
    """Return the internal (r,s) matrix added only on the a==b singles block.

    This is the piece of H_{(a r),(b s)} that is independent of the external orbital
    labels and is only present when a==b.

    Notes
    -----
    `dm2` is expected to use the `GUGAFCISolver.make_rdm12` convention:
      dm2[p,q,r,s] = <E_{p q} E_{r s}> - δ_{q r} <E_{p s}>.
    """

    internal = np.asarray(spaces.internal, dtype=np.int64).ravel()
    external = np.asarray(spaces.external, dtype=np.int64).ravel()
    if internal.size == 0:
        raise ValueError("spaces.internal must be non-empty")
    if int(np.min(internal)) != 0 or not bool(np.all(internal == np.arange(int(internal.size), dtype=np.int64))):
        raise NotImplementedError("ICRefSinglesRDM currently requires internal orbitals indexed as 0..nI-1")

    nI = int(internal.size)
    norb = int(h1e.shape[0])
    if np.any(external < 0) or np.any(external >= norb):
        raise ValueError("spaces.external indices out of range for h1e/eri4")

    h_int = h1e[:nI, :nI]
    eri_int = eri4[:nI, :nI, :nI, :nI]

    m = float(e_ref) * gamma
    m -= np.einsum("sq,rq->rs", h_int, gamma, optimize=True)
    m -= 0.5 * np.einsum("pqsu,pqru->rs", eri_int, dm2, optimize=True)
    m -= 0.5 * np.einsum("sqtu,rqtu->rs", eri_int, dm2, optimize=True)

    # Extra contraction-driven correction from external orbitals:
    #   -1/2 * Σ_{p,q internal} (Σ_{c in V} (p c| c q)) * dm2[r,s,p,q]
    if int(external.size):
        # Advanced indexing on both axes selects the diagonal (c,c) slice: eri[p,c,c,q] -> (p,c,q).
        j_ext = np.einsum("pcq->pq", eri4[:nI, external, external, :nI], optimize=True)
        dm2_rs_pq = dm2.transpose(2, 3, 0, 1)
        m += -0.5 * np.einsum("pq,rspq->rs", j_ext, dm2_rs_pq, optimize=True)

    return np.asarray(m, dtype=np.float64)


def _compute_h0_singles(
    *,
    h1e: np.ndarray,
    eri4: np.ndarray,
    gamma: np.ndarray,
    dm2: np.ndarray,
    singles: ICSingles,
    n_internal: int,
) -> np.ndarray:
    """Return H_{0,(a r)} for all singles labels in `singles`."""

    nI = int(n_internal)
    nlab = int(singles.nlab)
    if nlab == 0:
        return np.zeros(0, dtype=np.float64)

    norb = int(h1e.shape[0])
    a_all = np.asarray(singles.a, dtype=np.int64)
    r_all = np.asarray(singles.r, dtype=np.int64)
    if np.any(r_all < 0) or np.any(r_all >= nI):
        raise ValueError("singles.r out of range for internal RDMs")
    if np.any(a_all < 0) or np.any(a_all >= norb):
        raise ValueError("singles.a out of range for h1e/eri4")

    out = np.zeros(nlab, dtype=np.float64)
    order = np.asarray(singles.a_group_order, dtype=np.int64)
    offsets = np.asarray(singles.a_group_offsets, dtype=np.int64)
    keys = np.asarray(singles.a_group_keys, dtype=np.int64)

    for g, a in enumerate(keys.tolist()):
        start = int(offsets[g])
        stop = int(offsets[g + 1])
        if start == stop:
            continue
        idx = order[start:stop]
        r_idx = r_all[idx]

        h_pa = np.asarray(h1e[:nI, int(a)], dtype=np.float64)
        term1 = gamma.T @ h_pa  # r
        term2 = 0.5 * np.einsum("ptu,prtu->r", eri4[:nI, int(a), :nI, :nI], dm2, optimize=True)
        term3 = 0.5 * np.einsum("pqt,pqtr->r", eri4[:nI, :nI, :nI, int(a)], dm2, optimize=True)
        h0_a = term1 + term2 + term3

        out[idx] = h0_a[r_idx]

    return np.asarray(out, dtype=np.float64)


def _apply_hss_ref_singles(
    x: np.ndarray,
    *,
    h1e: np.ndarray,
    eri4: np.ndarray,
    gamma: np.ndarray,
    dm2: np.ndarray,
    singles: ICSingles,
    m_diag: np.ndarray,
    n_internal: int,
) -> np.ndarray:
    """Return y = H_ss x for the fully internally contracted singles block."""

    x = np.asarray(x, dtype=np.float64).ravel()
    if int(x.size) != int(singles.nlab):
        raise ValueError("x has wrong length for singles label set")
    if int(singles.nlab) == 0:
        return np.zeros(0, dtype=np.float64)

    nI = int(n_internal)

    r_all = np.asarray(singles.r, dtype=np.int64)

    order = np.asarray(singles.a_group_order, dtype=np.int64)
    offsets = np.asarray(singles.a_group_offsets, dtype=np.int64)
    keys = np.asarray(singles.a_group_keys, dtype=np.int64)
    n_groups = int(keys.size)

    # Build x as dense internal vectors per external group: X[g, r] = x_{a_g,r}.
    x_by_a = np.zeros((n_groups, nI), dtype=np.float64)
    for g in range(n_groups):
        start = int(offsets[g])
        stop = int(offsets[g + 1])
        if start == stop:
            continue
        idx = order[start:stop]
        x_by_a[g, r_all[idx]] = x[idx]

    y_by_a = np.zeros_like(x_by_a)

    # Dense pair loops (reference-oriented backend).
    for ia, a in enumerate(keys.tolist()):
        a = int(a)
        for ib, b in enumerate(keys.tolist()):
            b = int(b)
            xb = x_by_a[ib]
            if not np.any(xb):
                continue

            k = float(h1e[a, b]) * gamma
            k += 0.5 * np.einsum("tu,turs->rs", eri4[a, b, :nI, :nI], dm2, optimize=True)
            k += 0.5 * np.einsum("pq,pqrs->rs", eri4[:nI, :nI, a, b], dm2, optimize=True)
            k += 0.5 * np.einsum("qt,pqts->ps", eri4[a, :nI, :nI, b], dm2, optimize=True)
            if a == b:
                k += m_diag

            y_by_a[ia] += k @ xb

    y = np.zeros_like(x)
    for g in range(n_groups):
        start = int(offsets[g])
        stop = int(offsets[g + 1])
        if start == stop:
            continue
        idx = order[start:stop]
        y[idx] = y_by_a[g, r_all[idx]]
    return np.asarray(y, dtype=np.float64)


@dataclass
class ICRefSinglesRDM:
    """RDM/intermediate backend for the [reference + singles] contracted ic-MRCI space.

    Attributes
    ----------
    h1e : Any
        One-electron integrals.
    eri : Any
        Two-electron integrals.
    e_ref : float
        Reference energy.
    gamma : Any
        1-RDM.
    dm2 : Any
        2-RDM.
    spaces : OrbitalSpaces
        Orbital spaces.
    singles : ICSingles
        Singles labels.
    """

    h1e: Any
    eri: Any
    e_ref: float
    gamma: Any
    dm2: Any
    spaces: OrbitalSpaces
    singles: ICSingles

    _h1e: np.ndarray | None = None
    _eri4: np.ndarray | None = None
    _gamma: np.ndarray | None = None
    _dm2: np.ndarray | None = None
    _m_diag: np.ndarray | None = None
    _h0: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._h1e = _as_square_f64(self.h1e, name="h1e")
        norb = int(self._h1e.shape[0])
        self._eri4 = restore_eri1(self.eri, norb)

        self._gamma = _as_square_f64(self.gamma, name="gamma")
        self._dm2 = _as_dm2_f64(self.dm2)

        nI = int(self._gamma.shape[0])
        if int(self._gamma.shape[1]) != nI:
            raise ValueError("gamma must be square")
        if self._dm2.shape != (nI, nI, nI, nI):
            raise ValueError("dm2 shape must match gamma dimension")

        if norb < nI:
            raise ValueError("h1e/eri4 norb must be >= gamma dimension")

        self._m_diag = _compute_m_singles_diag(
            h1e=self._h1e,
            eri4=self._eri4,
            e_ref=float(self.e_ref),
            gamma=self._gamma,
            dm2=self._dm2,
            spaces=self.spaces,
        )
        self._h0 = _compute_h0_singles(
            h1e=self._h1e,
            eri4=self._eri4,
            gamma=self._gamma,
            dm2=self._dm2,
            singles=self.singles,
            n_internal=nI,
        )

    @property
    def nlab(self) -> int:
        return 1 + int(self.singles.nlab)

    def overlap(self, c: np.ndarray) -> np.ndarray:
        c = np.asarray(c, dtype=np.float64).ravel()
        if int(c.size) != int(self.nlab):
            raise ValueError("c has wrong length")
        c0 = float(c[0])
        cs = c[1:]
        rho0, rho_s = apply_overlap_ref_singles(c0=c0, c_singles=cs, singles=self.singles, gamma=self._gamma)  # type: ignore[arg-type]
        return np.concatenate(([rho0], rho_s))

    def sigma(self, c: np.ndarray) -> np.ndarray:
        c = np.asarray(c, dtype=np.float64).ravel()
        if int(c.size) != int(self.nlab):
            raise ValueError("c has wrong length")

        c0 = float(c[0])
        cs = np.asarray(c[1:], dtype=np.float64)

        h0 = np.asarray(self._h0, dtype=np.float64)
        sigma0 = float(self.e_ref) * c0 + float(np.dot(h0, cs))
        sigma_s = c0 * h0 + _apply_hss_ref_singles(
            cs,
            h1e=self._h1e,  # type: ignore[arg-type]
            eri4=self._eri4,  # type: ignore[arg-type]
            gamma=self._gamma,  # type: ignore[arg-type]
            dm2=self._dm2,  # type: ignore[arg-type]
            singles=self.singles,
            m_diag=self._m_diag,  # type: ignore[arg-type]
            n_internal=int(self._gamma.shape[0]),  # type: ignore[union-attr]
        )

        return np.concatenate(([sigma0], sigma_s))

    def solve(
        self,
        *,
        x0: np.ndarray | None = None,
        tol: float = 1e-10,
        max_cycle: int = 80,
        max_space: int = 25,
        s_tol: float = 1e-12,
    ) -> GeneralizedDavidsonResult:
        if x0 is None:
            x0 = np.zeros(self.nlab, dtype=np.float64)
            x0[0] = 1.0
        else:
            x0 = np.asarray(x0, dtype=np.float64).ravel()
            if int(x0.size) != int(self.nlab):
                raise ValueError("x0 has wrong length")

        return generalized_davidson1(
            self.sigma,
            self.overlap,
            x0,
            precond=None,
            tol=float(tol),
            max_cycle=int(max_cycle),
            max_space=int(max_space),
            s_tol=float(s_tol),
        )
