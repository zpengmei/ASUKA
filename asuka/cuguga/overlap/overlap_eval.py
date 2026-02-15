from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.oracle import _segment_value
from asuka.cuguga.overlap.overlap_segment import Generator, overlap_range, seg_code
from asuka.cuguga.overlap.overlap_tables import OverlapTables


def _path_nodes(drt: DRT, steps: np.ndarray) -> np.ndarray:
    node = int(drt.root)
    nodes = np.empty(int(drt.norb) + 1, dtype=np.int32)
    nodes[0] = node
    for k, s in enumerate(np.asarray(steps, dtype=np.int8).tolist()):
        node = int(drt.child[node, int(s)])
        if node < 0:
            raise ValueError("invalid path for this DRT")
        nodes[k + 1] = node
    return nodes


def _one_body_segment_q(gen: Generator, seg: int) -> str:
    if seg == gen.tail:
        return "uR" if gen.dir == "R" else "uL"
    if seg == gen.head:
        return "oR" if gen.dir == "R" else "oL"
    return "R" if gen.dir == "R" else "L"


@dataclass(frozen=True)
class OverlapResult:
    w0: float
    w1: float
    p2: float


def overlap_w0_w1(
    drt: DRT,
    bra_steps: np.ndarray,
    ket_steps: np.ndarray,
    gen1: Generator,
    gen2: Generator,
    tables: OverlapTables,
    *,
    mu: float = 0.0,
    star: bool = False,
    missing: set[tuple[str, int, int, int, int]] | None = None,
) -> OverlapResult:
    """Compute (w0, w1) overlap-channel products for ordered generators (gen1, gen2).

    This follows the overlap/non-overlap factorization used in Dobrautz' thesis:
      P2 = ∏_{p∈S2} W(...)
      w0 = P2 * ∏_{p∈S1} W^0(...)
      w1 = P2 * ∏_{p∈S1} W^1(...)

    Notes:
    - This routine does NOT decide the canonical operator ordering.
      You must pass generators in the intended order (gen1 then gen2).
    - For alike generators (RR/LL), `star=True` appends '*' to the seg_code used
      for overlap lookup (Table A.4 lower-sign convention).
    - This routine does not hard-code extra Δb restrictions; the tables control
      when W^0 / W^1 are nonzero for a given (Δb, d', d, b).
    """

    norb = int(drt.norb)
    bra_steps = np.asarray(bra_steps, dtype=np.int8).ravel()
    ket_steps = np.asarray(ket_steps, dtype=np.int8).ravel()
    if bra_steps.size != norb or ket_steps.size != norb:
        raise ValueError("steps have wrong length for this DRT")

    # For a nonzero matrix element, the bra/ket paths must be identical outside
    # the union of the two generator ranges. (GUGA loop locality: generators do
    # not alter coupling outside their excitation ranges.)
    a1, b1 = int(gen1.tail), int(gen1.head)
    a2, b2 = int(gen2.tail), int(gen2.head)
    union_start = min(a1, a2)
    union_end = max(b1, b2)
    if union_start > 0 and not np.array_equal(bra_steps[:union_start], ket_steps[:union_start]):
        return OverlapResult(w0=0.0, w1=0.0, p2=0.0)
    if union_end + 1 < norb and not np.array_equal(bra_steps[union_end + 1 :], ket_steps[union_end + 1 :]):
        return OverlapResult(w0=0.0, w1=0.0, p2=0.0)
    # If generator ranges are disjoint, the interior gap is also outside the union.
    if b1 < a2 - 1 and not np.array_equal(bra_steps[b1 + 1 : a2], ket_steps[b1 + 1 : a2]):
        return OverlapResult(w0=0.0, w1=0.0, p2=0.0)
    if b2 < a1 - 1 and not np.array_equal(bra_steps[b2 + 1 : a1], ket_steps[b2 + 1 : a1]):
        return OverlapResult(w0=0.0, w1=0.0, p2=0.0)

    bra_nodes = _path_nodes(drt, bra_steps)
    ket_nodes = _path_nodes(drt, ket_steps)

    ov = overlap_range(gen1, gen2)
    has_overlap = ov.start != ov.stop

    ov_start = int(ov.start)
    ov_end = int(ov.stop) - 1

    b_ket_all = drt.node_twos[ket_nodes[1:]].astype(np.int32, copy=False)
    b_bra_all = drt.node_twos[bra_nodes[1:]].astype(np.int32, copy=False)

    p2 = 1.0
    for seg in range(union_start, union_end + 1):
        if not ((a1 <= seg <= b1) or (a2 <= seg <= b2)):
            continue
        if has_overlap and ov_start <= seg <= ov_end:
            continue
        active = gen1 if (a1 <= seg <= b1) else gen2
        q = _one_body_segment_q(active, seg)
        d = int(ket_steps[seg])
        dprime = int(bra_steps[seg])
        b = int(b_ket_all[seg])
        db = int(b - int(b_bra_all[seg]))
        w = _segment_value(q, dprime, d, db, b)
        if w == 0.0:
            return OverlapResult(w0=0.0, w1=0.0, p2=0.0)
        p2 *= float(w)

    if not has_overlap:
        return OverlapResult(w0=float(p2), w1=0.0, p2=float(p2))

    w0 = 1.0
    w1 = 1.0
    for seg in range(ov_start, ov_end + 1):
        d = int(ket_steps[seg])
        dprime = int(bra_steps[seg])
        b = int(b_ket_all[seg])
        db = int(b - int(b_bra_all[seg]))

        code = seg_code(gen1, gen2, seg)
        if star and gen1.dir == gen2.dir:
            code = f"{code}*"

        if missing is not None:
            if tables.get_expr(code, db, 0, dprime, d) is None:
                missing.add((code, int(db), 0, int(dprime), int(d)))
            if tables.get_expr(code, db, 1, dprime, d) is None:
                missing.add((code, int(db), 1, int(dprime), int(d)))

        if w0 != 0.0:
            v0 = tables.value(code, db, 0, dprime, d, b=b, mu=mu)
            w0 = 0.0 if v0 == 0.0 else (w0 * float(v0))

        if w1 != 0.0:
            v1 = tables.value(code, db, 1, dprime, d, b=b, mu=mu)
            w1 = 0.0 if v1 == 0.0 else (w1 * float(v1))

    return OverlapResult(w0=float(p2 * w0), w1=float(p2 * w1), p2=float(p2))
