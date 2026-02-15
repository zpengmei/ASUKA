"""Two-body coupling via overlap-table evaluator (Tables A.4/A.5).

This module is an incremental replacement for the `E_pq E_rs` expansion used in
`oracle.connected_row`.

The oracle implements the 2-body part as:
  H2 = 1/2 * Σ_{pqrs} (pq|rs) E_pq E_rs

For proper doubles (4 distinct orbitals), only four (p,q,r,s) tuples can connect
|ket> -> |bra|: 2 particle/hole matchings × 2 operator orders. We evaluate each
required matrix element <bra|E_pq E_rs|ket> via the overlap-table machinery.
When the excitation ranges do not overlap (S1 empty), the result reduces to a
product of non-overlap segment factors and no overlap-table lookups are needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asuka.cuguga.overlap.excitation_classifier import ExcitationInfo, classify_excitation
from asuka.cuguga.overlap.overlap_eval import overlap_w0_w1
from asuka.cuguga.overlap.overlap_segment import Generator
from asuka.cuguga.overlap.overlap_tables import OverlapTables


def _epq_epq_matrix_element(
    drt,
    bra_idx: int,
    ket_idx: int,
    p: int,
    q: int,
    r: int,
    s: int,
    tables: OverlapTables,
    *,
    mu: float,
    missing: set[tuple[str, int, int, int, int]] | None,
) -> tuple[float, float, float, float]:
    """Return (<bra|E_pq E_rs|ket>, w0, w1) for a specific (p,q,r,s).

    - If the excitation ranges overlap (S1 non-empty), (w0,w1) are the overlap-channel
      products and the matrix element is (w0+w1).
    - If S1 is empty, `overlap_w0_w1` returns a single-channel result (w1=0) computed
      from non-overlap segment factors (no overlap-table lookups).
    """

    bra_steps = drt.index_to_path(int(bra_idx))
    ket_steps = drt.index_to_path(int(ket_idx))
    gen1 = Generator(int(p), int(q))
    gen2 = Generator(int(r), int(s))

    # Mixed generators (RL/LR): Table A.5 states the contribution sign is independent of
    # generator order. Canonicalize to (R then L) so we only need to store RL-type
    # overlap segment values.
    if gen1.dir != gen2.dir and gen1.dir == "L" and gen2.dir == "R":
        gen1, gen2 = gen2, gen1

    ovr = overlap_w0_w1(
        drt,
        bra_steps,
        ket_steps,
        gen1,
        gen2,
        tables,
        mu=float(mu),
        star=False,
        missing=missing,
    )
    w0 = float(ovr.w0)
    w1 = float(ovr.w1)
    return float(w0 + w1), w0, w1, float(ovr.p2)


@dataclass(frozen=True)
class TwoBodyEvalResult:
    h2: float
    w0: float
    w1: float
    p2: float
    info: ExcitationInfo


def two_body_h2_via_overlap(
    drt,
    bra_idx: int,
    ket_idx: int,
    eri4: np.ndarray,
    tables: OverlapTables,
    *,
    mu: float = 0.0,
    missing: set[tuple[str, int, int, int, int]] | None = None,
    include_global_half: bool = False,
) -> TwoBodyEvalResult:
    """Compute H2(bra, ket) for proper doubles using overlap tables.

    Parameters
    ----------
    include_global_half:
        If True, multiply the final value by an *additional* 1/2.
        The base implementation already includes the 1/2 prefactor that appears
        in `oracle.connected_row` for the (pq|rs)E_pqE_rs term.
    """

    info = classify_excitation(drt, bra_idx, ket_idx)
    if not info.is_double:
        return TwoBodyEvalResult(h2=0.0, w0=0.0, w1=0.0, p2=1.0, info=info)

    eri4 = np.asarray(eri4, dtype=np.float64)
    if eri4.ndim != 4:
        raise ValueError("eri4 must be a 4-index tensor eri[p,q,r,s]")

    p1, p2 = info.particles
    h1, h2 = info.holes

    # Two matchings, two orders each (4 terms total).
    pqrs_terms = [
        (p1, h1, p2, h2),
        (p2, h2, p1, h1),
        (p1, h2, p2, h1),
        (p2, h1, p1, h2),
    ]

    # Track one representative (w0,w1) for debugging output.
    dbg_w0 = 0.0
    dbg_w1 = 0.0
    dbg_p2 = 1.0

    h2_val = 0.0
    for p, q, r, s in pqrs_terms:
        me, w0, w1, p2fac = _epq_epq_matrix_element(
            drt,
            bra_idx,
            ket_idx,
            int(p),
            int(q),
            int(r),
            int(s),
            tables,
            mu=float(mu),
            missing=missing,
        )

        if info.gen1 is not None and info.gen2 is not None:
            if (int(p), int(q), int(r), int(s)) == (
                int(info.gen1.p),
                int(info.gen1.q),
                int(info.gen2.p),
                int(info.gen2.q),
            ):
                dbg_w0, dbg_w1, dbg_p2 = float(w0), float(w1), float(p2fac)

        # Match oracle's 1/2 prefactor on (pq|rs) E_pq E_rs.
        h2_val += 0.5 * float(eri4[int(p), int(q), int(r), int(s)]) * float(me)

    if include_global_half:
        h2_val *= 0.5

    return TwoBodyEvalResult(h2=float(h2_val), w0=dbg_w0, w1=dbg_w1, p2=dbg_p2, info=info)
