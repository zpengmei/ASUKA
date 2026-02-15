"""Occupancy-driven excitation classifier for CSF (bra, ket) pairs.

This is a small helper for the overlap-table implementation.

Given two CSF indices (bra, ket), we compute Δocc = occ_bra - occ_ket and classify:
  - diag: bra == ket
  - single: 2 orbitals change with Δocc = (+1, -1)
  - double_like: 4 orbitals change with Δocc = (+1,+1,-1,-1) and particles/holes are separated
  - double_mixed: 4 orbitals change with Δocc = (+1,+1,-1,-1) and particles/holes are interleaved
  - same_occ: same occupation pattern but different CSF (spin recoupling)
  - other: everything else

For proper doubles it also returns a canonical *overlapping* generator pair (gen1, gen2)
to feed into `overlap_w0_w1(...)` and Appendix-B style ERI combinations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from asuka.cuguga.overlap.overlap_segment import Generator

ExcFamily = Literal["diag", "single", "double_like", "double_mixed", "same_occ", "other"]


@dataclass(frozen=True)
class ExcitationInfo:
    family: ExcFamily
    changed: tuple[int, ...]
    particles: tuple[int, ...]
    holes: tuple[int, ...]
    ijkl: tuple[int, int, int, int] | tuple[()]
    pattern: str
    gen1: Optional[Generator]
    gen2: Optional[Generator]

    @property
    def is_double(self) -> bool:
        return self.family in ("double_like", "double_mixed")

    @property
    def is_like(self) -> bool:
        return self.family == "double_like"

    @property
    def is_mixed(self) -> bool:
        return self.family == "double_mixed"


def _delta_occ(occ_bra: Sequence[int], occ_ket: Sequence[int]) -> tuple[int, ...]:
    if len(occ_bra) != len(occ_ket):
        raise ValueError("occ_bra and occ_ket must have the same length")
    return tuple(int(a - b) for a, b in zip(occ_bra, occ_ket))


def classify_excitation(drt, bra_idx: int, ket_idx: int) -> ExcitationInfo:
    """Classify the excitation connecting |ket> -> |bra> by Δocc."""

    bra_idx = int(bra_idx)
    ket_idx = int(ket_idx)
    if bra_idx == ket_idx:
        return ExcitationInfo("diag", (), (), (), (), "", None, None)

    bra_steps = drt.index_to_path(bra_idx)
    ket_steps = drt.index_to_path(ket_idx)
    occ_bra = drt.path_to_occ(bra_steps)
    occ_ket = drt.path_to_occ(ket_steps)

    delta = _delta_occ(occ_bra, occ_ket)
    changed = tuple(i for i, d in enumerate(delta) if d != 0)

    if len(changed) == 0:
        return ExcitationInfo("same_occ", (), (), (), (), "", None, None)

    if len(changed) == 2:
        particles = tuple(i for i in changed if delta[i] == +1)
        holes = tuple(i for i in changed if delta[i] == -1)
        if len(particles) == 1 and len(holes) == 1:
            return ExcitationInfo("single", changed, particles, holes, (), "", None, None)
        return ExcitationInfo("other", changed, (), (), (), "", None, None)

    if len(changed) == 4:
        particles = tuple(sorted(i for i in changed if delta[i] == +1))
        holes = tuple(sorted(i for i in changed if delta[i] == -1))
        if len(particles) != 2 or len(holes) != 2:
            return ExcitationInfo("other", changed, particles, holes, (), "", None, None)

        ijkl: tuple[int, int, int, int] = tuple(sorted(changed))  # type: ignore[assignment]
        pattern = "".join("+" if delta[i] == +1 else "-" for i in ijkl)

        p_low, p_high = particles
        h_low, h_high = holes
        like = (p_high < h_low) or (h_high < p_low)

        if like:
            # Canonical overlapping operator for like case:
            #   e_{p_high,h_high ; p_low,h_low}
            gen1 = Generator(p_high, h_high)
            gen2 = Generator(p_low, h_low)
            family: ExcFamily = "double_like"
        else:
            # Canonical overlapping operator for mixed case:
            #   e_{p_low,h_high ; p_high,h_low}
            gen1 = Generator(p_low, h_high)
            gen2 = Generator(p_high, h_low)
            family = "double_mixed"

        return ExcitationInfo(family, changed, particles, holes, ijkl, pattern, gen1, gen2)

    return ExcitationInfo("other", changed, (), (), (), "", None, None)
