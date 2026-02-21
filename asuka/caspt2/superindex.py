"""Superindex infrastructure for the 13-case internally contracted CASPT2 basis.

Ports OpenMolcas ``superindex.f`` (``SUPINI``) for C1 symmetry.
Each IC case pairs an *active superindex* (functions of active-space indices)
with an *inactive/virtual superindex* (functions of inactive/virtual indices).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CASOrbitals:
    """Orbital-space dimensions."""

    nfro: int  # frozen core (0 for now)
    nish: int  # inactive (doubly occupied, non-frozen)
    nash: int  # active
    nssh: int  # secondary (virtual)

    @property
    def nmo(self) -> int:
        return self.nfro + self.nish + self.nash + self.nssh

    @property
    def nocc(self) -> int:
        return self.nfro + self.nish + self.nash

    @property
    def ncore(self) -> int:
        return self.nfro + self.nish


@dataclass(frozen=True)
class SuperindexMap:
    """Precomputed index mappings for all 13 IC cases (C1 only)."""

    orbs: CASOrbitals

    # Active triple indices: t,u,v all in [0, nash)
    ktuv: np.ndarray   # (nash, nash, nash) -> superindex (-1 if unused)
    mtuv: np.ndarray   # (ntuv, 3) -> (t, u, v)
    ntuv: int

    # Active pair indices: all tu pairs
    ktu: np.ndarray    # (nash, nash) -> superindex
    mtu: np.ndarray    # (ntu, 2) -> (t, u)
    ntu: int

    # Symmetric pairs: t >= u
    ktgeu: np.ndarray
    mtgeu: np.ndarray
    ntgeu: int

    # Antisymmetric pairs: t > u
    ktgtu: np.ndarray
    mtgtu: np.ndarray
    ntgtu: int

    # Inactive pairs: i >= j
    kigej: np.ndarray
    migej: np.ndarray
    nigej: int

    # Inactive pairs: i > j
    kigtj: np.ndarray
    migtj: np.ndarray
    nigtj: int

    # Virtual pairs: a >= b
    kageb: np.ndarray
    mageb: np.ndarray
    nageb: int

    # Virtual pairs: a > b
    kagtb: np.ndarray
    magtb: np.ndarray
    nagtb: int

    # Per-case dimensions
    nasup: np.ndarray   # (13,) active superindex dimension per case
    nisup: np.ndarray   # (13,) inactive/virtual superindex dimension per case
    nindep: np.ndarray  # (13,) filled after linear-dep removal


def _build_triple_indices(nash: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Build active triple-index mapping: all (t, u, v) triples."""
    ktuv = np.full((nash, nash, nash), -1, dtype=np.int64)
    triples = []
    idx = 0
    for t in range(nash):
        for u in range(nash):
            for v in range(nash):
                ktuv[t, u, v] = idx
                triples.append((t, u, v))
                idx += 1
    ntuv = idx
    mtuv = np.array(triples, dtype=np.int64).reshape(ntuv, 3) if ntuv > 0 else np.empty((0, 3), dtype=np.int64)
    return ktuv, mtuv, ntuv


def _build_pair_indices(n: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Build all pair indices (p, q) for p, q in [0, n)."""
    k = np.full((n, n), -1, dtype=np.int64)
    pairs = []
    idx = 0
    for p in range(n):
        for q in range(n):
            k[p, q] = idx
            pairs.append((p, q))
            idx += 1
    m = np.array(pairs, dtype=np.int64).reshape(idx, 2) if idx > 0 else np.empty((0, 2), dtype=np.int64)
    return k, m, idx


def _build_sym_pair_indices(n: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Build symmetric pair indices: p >= q."""
    k = np.full((n, n), -1, dtype=np.int64)
    pairs = []
    idx = 0
    for p in range(n):
        for q in range(p + 1):
            k[p, q] = idx
            pairs.append((p, q))
            idx += 1
    m = np.array(pairs, dtype=np.int64).reshape(idx, 2) if idx > 0 else np.empty((0, 2), dtype=np.int64)
    return k, m, idx


def _build_asym_pair_indices(n: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Build antisymmetric pair indices: p > q."""
    k = np.full((n, n), -1, dtype=np.int64)
    pairs = []
    idx = 0
    for p in range(n):
        for q in range(p):
            k[p, q] = idx
            pairs.append((p, q))
            idx += 1
    m = np.array(pairs, dtype=np.int64).reshape(idx, 2) if idx > 0 else np.empty((0, 2), dtype=np.int64)
    return k, m, idx


def build_superindex(nish: int, nash: int, nssh: int) -> SuperindexMap:
    """Build the complete superindex map for all 13 IC cases (C1 symmetry).

    Parameters
    ----------
    nish : int
        Number of inactive (doubly occupied) orbitals.
    nash : int
        Number of active orbitals.
    nssh : int
        Number of secondary (virtual) orbitals.
    """
    orbs = CASOrbitals(nfro=0, nish=nish, nash=nash, nssh=nssh)

    # Active triples
    ktuv, mtuv, ntuv = _build_triple_indices(nash)
    # Active pairs
    ktu, mtu, ntu = _build_pair_indices(nash)
    ktgeu, mtgeu, ntgeu = _build_sym_pair_indices(nash)
    ktgtu, mtgtu, ntgtu = _build_asym_pair_indices(nash)
    # Inactive pairs
    kigej, migej, nigej = _build_sym_pair_indices(nish)
    kigtj, migtj, nigtj = _build_asym_pair_indices(nish)
    # Virtual pairs
    kageb, mageb, nageb = _build_sym_pair_indices(nssh)
    kagtb, magtb, nagtb = _build_asym_pair_indices(nssh)

    # Per-case dimensions (NASUP, NISUP)
    nasup = np.zeros(13, dtype=np.int64)
    nisup = np.zeros(13, dtype=np.int64)

    # Case 1 (A): VJTU - 3 active + 1 inactive
    nasup[0] = ntuv
    nisup[0] = nish
    # Case 2 (B+): VJTIP - 2 active(sym) + 2 inactive(sym)
    nasup[1] = ntgeu
    nisup[1] = nigej
    # Case 3 (B-): VJTIM - 2 active(asym) + 2 inactive(asym)
    nasup[2] = ntgtu
    nisup[2] = nigtj
    # Case 4 (C): ATVX - 3 active + 1 virtual
    nasup[3] = ntuv
    nisup[3] = nssh
    # Case 5 (D): AIVX - mixed ia coupling
    nasup[4] = 2 * ntu
    nisup[4] = nssh * nish
    # Case 6 (E+): VJAIP - 1 active + (virtual, sym inactive pair)
    #   ext_idx = a + nssh * igej, where igej indexes i>=j pairs
    nasup[5] = nash
    nisup[5] = nssh * nigej
    # Case 7 (E-): VJAIM - 1 active + (virtual, asym inactive pair)
    #   ext_idx = a + nssh * igtj, where igtj indexes i>j pairs
    nasup[6] = nash
    nisup[6] = nssh * nigtj
    # Case 8 (F+): BVATP - 2 active(sym) + 2 virtual(sym)
    nasup[7] = ntgeu
    nisup[7] = nageb
    # Case 9 (F-): BVATM - 2 active(asym) + 2 virtual(asym)
    nasup[8] = ntgtu
    nisup[8] = nagtb
    # Case 10 (G+): BJATQ - 1 active + (inactive, sym virtual pair)
    #   ext_idx = i + nish * iageb, where iageb indexes a>=b pairs
    nasup[9] = nash
    nisup[9] = nish * nageb
    # Case 11 (G-): BJATM - 1 active + (inactive, asym virtual pair)
    #   ext_idx = i + nish * iagtb, where iagtb indexes a>b pairs
    nasup[10] = nash
    nisup[10] = nish * nagtb
    # Case 12 (H+): BJAIP - 2 virtual(sym) + 2 inactive(sym)
    nasup[11] = nageb
    nisup[11] = nigej
    # Case 13 (H-): BJAIM - 2 virtual(asym) + 2 inactive(asym)
    nasup[12] = nagtb
    nisup[12] = nigtj

    nindep = nasup.copy()  # will be updated after linear-dep removal

    return SuperindexMap(
        orbs=orbs,
        ktuv=ktuv, mtuv=mtuv, ntuv=ntuv,
        ktu=ktu, mtu=mtu, ntu=ntu,
        ktgeu=ktgeu, mtgeu=mtgeu, ntgeu=ntgeu,
        ktgtu=ktgtu, mtgtu=mtgtu, ntgtu=ntgtu,
        kigej=kigej, migej=migej, nigej=nigej,
        kigtj=kigtj, migtj=migtj, nigtj=nigtj,
        kageb=kageb, mageb=mageb, nageb=nageb,
        kagtb=kagtb, magtb=magtb, nagtb=nagtb,
        nasup=nasup, nisup=nisup, nindep=nindep,
    )
