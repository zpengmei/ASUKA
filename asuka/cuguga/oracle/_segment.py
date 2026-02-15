"""Segment value computation for single-generator GUGA matrix elements.

This module implements the segment value functions from Dobrautz et al.,
Table 1 (tab:single-mateles). Segment values are the fundamental building
blocks for computing matrix elements of single-electron generators E_pq
in the GUGA/CSF framework.

Two evaluation strategies are provided:

1. **LUT-based** (default for b <= 4096): Precomputed lookup tables indexed
   by a "value code" that selects the appropriate b-dependent function.
   This avoids branchy dispatch in hot loops.

2. **Fallback** (b > 4096): Direct evaluation using the original branchy
   formulas involving sqrt and reciprocal functions.

References
----------
W. Dobrautz, PhD thesis, University of Stuttgart, 2019.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Helper functions A(b, x, y) and C(b, x) from the GUGA segment formulas
# ---------------------------------------------------------------------------


def _A(b: int, x: int, y: int) -> float:
    """Compute A(b, x, y) = sqrt((b + x) / (b + y))."""
    return math.sqrt((b + x) / (b + y))


def _C(b: int, x: int) -> float:
    """Compute C(b, x) = sqrt((b + x - 1)(b + x + 1)) / (b + x)."""
    return math.sqrt((b + x - 1) * (b + x + 1)) / (b + x)


# ---------------------------------------------------------------------------
# Integer-encoded segment labels (avoid string dispatch in hot loops)
# ---------------------------------------------------------------------------

_Q_W = 0
_Q_uR = 1
_Q_R = 2
_Q_oR = 3
_Q_uL = 4
_Q_L = 5
_Q_oL = 6

_Q_FROM_STR: dict[str, int] = {
    "W": _Q_W,
    "uR": _Q_uR,
    "R": _Q_R,
    "oR": _Q_oR,
    "uL": _Q_uL,
    "L": _Q_L,
    "oL": _Q_oL,
}

# ---------------------------------------------------------------------------
# Segment value codes and LUT infrastructure
# ---------------------------------------------------------------------------

_SEG_LUT_MAX_B = 4096

# Segment value "codes" (indexes into ``_SV_BY_CODE``).
_SV_ZERO = 0
_SV_ONE = 1
_SV_TWO = 2
_SV_NEG_ONE = 3
_SV_A01 = 4
_SV_A10 = 5
_SV_A12 = 6
_SV_A21 = 7
_SV_INV_B = 8
_SV_INV_BP1 = 9
_SV_INV_BP2 = 10
_SV_NEG_INV_BP1 = 11
_SV_NEG_INV_BP2 = 12
_SV_C0 = 13
_SV_C1 = 14
_SV_C2 = 15


def _build_segment_value_code_arrays(max_b: int) -> tuple[list[float], ...]:
    """Build precomputed value arrays for each segment value code.

    Parameters
    ----------
    max_b : int
        Maximum b value to precompute for.

    Returns
    -------
    tuple of list[float]
        Tuple of 16 lists, one per segment value code, each of length
        ``max_b + 1``. Index with ``_SV_BY_CODE[code][b]``.
    """
    size = max_b + 1

    zero = [0.0] * size
    one = [1.0] * size
    two = [2.0] * size
    neg_one = [-1.0] * size

    inv_b = [0.0] * size
    for b in range(1, size):
        inv_b[b] = 1.0 / b

    inv_bp1 = [1.0 / (b + 1) for b in range(size)]
    inv_bp2 = [1.0 / (b + 2) for b in range(size)]
    neg_inv_bp1 = [-x for x in inv_bp1]
    neg_inv_bp2 = [-x for x in inv_bp2]

    # A(b,x,y) = sqrt((b+x)/(b+y))
    a01 = [math.sqrt(b / (b + 1)) for b in range(size)]
    a10 = [float("nan")] * size
    for b in range(1, size):
        a10[b] = 1.0 / a01[b]
    a12 = [math.sqrt((b + 1) / (b + 2)) for b in range(size)]
    a21 = [1.0 / x for x in a12]

    # C(b,x) = sqrt((b+x-1)(b+x+1))/(b+x)
    c0 = [float("nan")] * size
    for b in range(1, size):
        c0[b] = math.sqrt((b - 1) * (b + 1)) / b
    c1 = [math.sqrt(b * (b + 2)) / (b + 1) for b in range(size)]
    c2 = [math.sqrt((b + 1) * (b + 3)) / (b + 2) for b in range(size)]

    return (
        zero,
        one,
        two,
        neg_one,
        a01,
        a10,
        a12,
        a21,
        inv_b,
        inv_bp1,
        inv_bp2,
        neg_inv_bp1,
        neg_inv_bp2,
        c0,
        c1,
        c2,
    )


_SV_BY_CODE = _build_segment_value_code_arrays(_SEG_LUT_MAX_B)

# LUTs return a value code id; the final value is ``_SV_BY_CODE[code][b]``.
_SV_LUT_W: tuple[tuple[int, int, int, int], ...] = (
    (_SV_ZERO, _SV_ZERO, _SV_ZERO, _SV_ZERO),
    (_SV_ZERO, _SV_ONE, _SV_ZERO, _SV_ZERO),
    (_SV_ZERO, _SV_ZERO, _SV_ONE, _SV_ZERO),
    (_SV_ZERO, _SV_ZERO, _SV_ZERO, _SV_TWO),
)
_SV_LUT_uR: tuple[tuple[int, int, int, int], ...] = (
    (_SV_ZERO, _SV_ZERO, _SV_ZERO, _SV_ZERO),
    (_SV_ONE, _SV_ZERO, _SV_ZERO, _SV_ZERO),
    (_SV_ONE, _SV_ZERO, _SV_ZERO, _SV_ZERO),
    (_SV_ZERO, _SV_A10, _SV_A12, _SV_ZERO),
)
_SV_LUT_oR: tuple[tuple[int, int, int, int], ...] = (
    (_SV_ZERO, _SV_ONE, _SV_ONE, _SV_ZERO),
    (_SV_ZERO, _SV_ZERO, _SV_ZERO, _SV_A01),
    (_SV_ZERO, _SV_ZERO, _SV_ZERO, _SV_A21),
    (_SV_ZERO, _SV_ZERO, _SV_ZERO, _SV_ZERO),
)
_SV_LUT_uL: tuple[tuple[int, int, int, int], ...] = (
    (_SV_ZERO, _SV_ONE, _SV_ONE, _SV_ZERO),
    (_SV_ZERO, _SV_ZERO, _SV_ZERO, _SV_A21),
    (_SV_ZERO, _SV_ZERO, _SV_ZERO, _SV_A01),
    (_SV_ZERO, _SV_ZERO, _SV_ZERO, _SV_ZERO),
)
_SV_LUT_oL: tuple[tuple[int, int, int, int], ...] = (
    (_SV_ZERO, _SV_ZERO, _SV_ZERO, _SV_ZERO),
    (_SV_ONE, _SV_ZERO, _SV_ZERO, _SV_ZERO),
    (_SV_ONE, _SV_ZERO, _SV_ZERO, _SV_ZERO),
    (_SV_ZERO, _SV_A01, _SV_A21, _SV_ZERO),
)

_SV_LUT_R_DBM1: tuple[tuple[int, int, int, int], ...] = (
    (_SV_ONE, _SV_ZERO, _SV_ZERO, _SV_ZERO),
    (_SV_ZERO, _SV_NEG_ONE, _SV_NEG_INV_BP2, _SV_ZERO),
    (_SV_ZERO, _SV_ZERO, _SV_C2, _SV_ZERO),
    (_SV_ZERO, _SV_ZERO, _SV_ZERO, _SV_NEG_ONE),
)
_SV_LUT_R_DBP1: tuple[tuple[int, int, int, int], ...] = (
    (_SV_ONE, _SV_ZERO, _SV_ZERO, _SV_ZERO),
    (_SV_ZERO, _SV_C0, _SV_ZERO, _SV_ZERO),
    (_SV_ZERO, _SV_INV_B, _SV_NEG_ONE, _SV_ZERO),
    (_SV_ZERO, _SV_ZERO, _SV_ZERO, _SV_NEG_ONE),
)
_SV_LUT_L_DBM1: tuple[tuple[int, int, int, int], ...] = (
    (_SV_ONE, _SV_ZERO, _SV_ZERO, _SV_ZERO),
    (_SV_ZERO, _SV_C1, _SV_INV_BP1, _SV_ZERO),
    (_SV_ZERO, _SV_ZERO, _SV_NEG_ONE, _SV_ZERO),
    (_SV_ZERO, _SV_ZERO, _SV_ZERO, _SV_NEG_ONE),
)
_SV_LUT_L_DBP1: tuple[tuple[int, int, int, int], ...] = (
    (_SV_ONE, _SV_ZERO, _SV_ZERO, _SV_ZERO),
    (_SV_ZERO, _SV_NEG_ONE, _SV_ZERO, _SV_ZERO),
    (_SV_ZERO, _SV_NEG_INV_BP1, _SV_C1, _SV_ZERO),
    (_SV_ZERO, _SV_ZERO, _SV_ZERO, _SV_NEG_ONE),
)


# ---------------------------------------------------------------------------
# Segment value evaluation functions
# ---------------------------------------------------------------------------


def _segment_value_int_fallback(
    q: int,
    dprime: int,
    d: int,
    db: int,
    b: int,
) -> float:
    """Single-generator segment values from Dobrautz et al. Table~1 (tab:single-mateles).

    This is a hot-loop variant of :func:`_segment_value` that assumes all inputs are Python
    ints and uses an integer-encoded segment label `q` to avoid string dispatch overhead.
    """

    if q == _Q_W:
        if dprime != d:
            return 0.0
        if d == 0:
            return 0.0
        if d in (1, 2):
            return 1.0
        if d == 3:
            return 2.0
        return 0.0

    if q == _Q_uR:
        if d == 0 and dprime in (1, 2):
            return 1.0
        if dprime == 3 and d == 1:
            return _A(b, 1, 0)
        if dprime == 3 and d == 2:
            return _A(b, 1, 2)
        return 0.0

    if q == _Q_oR:
        if dprime == 0 and d in (1, 2):
            return 1.0
        if dprime == 1 and d == 3:
            return _A(b, 0, 1)
        if dprime == 2 and d == 3:
            return _A(b, 2, 1)
        return 0.0

    if q == _Q_uL:
        if dprime == 0 and d in (1, 2):
            return 1.0
        if dprime == 1 and d == 3:
            return _A(b, 2, 1)
        if dprime == 2 and d == 3:
            return _A(b, 0, 1)
        return 0.0

    if q == _Q_oL:
        if d == 0 and dprime in (1, 2):
            return 1.0
        if dprime == 3 and d == 1:
            return _A(b, 0, 1)
        if dprime == 3 and d == 2:
            return _A(b, 2, 1)
        return 0.0

    if q == _Q_R:
        if db not in (-1, +1):
            return 0.0
        if dprime == 0 and d == 0:
            return 1.0
        if dprime == 1 and d == 1:
            return -1.0 if db == -1 else _C(b, 0)
        if dprime == 1 and d == 2:
            return -1.0 / (b + 2) if db == -1 else 0.0
        if dprime == 2 and d == 1:
            return 0.0 if db == -1 else (1.0 / b if b != 0 else 0.0)
        if dprime == 2 and d == 2:
            return _C(b, 2) if db == -1 else -1.0
        if dprime == 3 and d == 3:
            return -1.0
        return 0.0

    if q == _Q_L:
        if db not in (-1, +1):
            return 0.0
        if dprime == 0 and d == 0:
            return 1.0
        if dprime == 1 and d == 1:
            return _C(b, 1) if db == -1 else -1.0
        if dprime == 1 and d == 2:
            return 1.0 / (b + 1) if db == -1 else 0.0
        if dprime == 2 and d == 1:
            return 0.0 if db == -1 else -1.0 / (b + 1)
        if dprime == 2 and d == 2:
            return -1.0 if db == -1 else _C(b, 1)
        if dprime == 3 and d == 3:
            return -1.0
        return 0.0

    raise ValueError(f"unknown segment type {q!r}")


def _segment_value_int(
    q: int,
    dprime: int,
    d: int,
    db: int,
    b: int,
) -> float:
    """Single-generator segment values from Dobrautz et al. Table~1 (tab:single-mateles).

    This is a hot-loop variant of :func:`_segment_value` that assumes all inputs are Python
    ints and uses an integer-encoded segment label `q` to avoid string dispatch overhead.

    Implementation note: for common cases we use a LUT that maps ``(q, d', d [, sign(db)])``
    to a precomputed ``b``-dependent value, falling back to the original branchy
    implementation when ``b > _SEG_LUT_MAX_B``.

    Parameters
    ----------
    q : int
        Integer-encoded segment label (one of ``_Q_W``, ``_Q_uR``, etc.).
    dprime : int
        Bra step value at this segment (0=E, 1=U, 2=L, 3=D).
    d : int
        Ket step value at this segment.
    db : int
        Spin difference b_ket - b_bra at the child node.
    b : int
        Spin quantum number (2S) at the child node of the ket path.

    Returns
    -------
    float
        The segment value W(q; d', d; db, b).
    """

    if b <= _SEG_LUT_MAX_B:
        if q == _Q_W:
            return _SV_BY_CODE[_SV_LUT_W[dprime][d]][b]
        if q == _Q_uR:
            return _SV_BY_CODE[_SV_LUT_uR[dprime][d]][b]
        if q == _Q_oR:
            return _SV_BY_CODE[_SV_LUT_oR[dprime][d]][b]
        if q == _Q_uL:
            return _SV_BY_CODE[_SV_LUT_uL[dprime][d]][b]
        if q == _Q_oL:
            return _SV_BY_CODE[_SV_LUT_oL[dprime][d]][b]
        if q == _Q_R:
            if db == -1:
                return _SV_BY_CODE[_SV_LUT_R_DBM1[dprime][d]][b]
            if db == +1:
                return _SV_BY_CODE[_SV_LUT_R_DBP1[dprime][d]][b]
            return 0.0
        if q == _Q_L:
            if db == -1:
                return _SV_BY_CODE[_SV_LUT_L_DBM1[dprime][d]][b]
            if db == +1:
                return _SV_BY_CODE[_SV_LUT_L_DBP1[dprime][d]][b]
            return 0.0

    return _segment_value_int_fallback(q, dprime, d, db, b)


def _segment_value(
    q: str,
    dprime: int,
    d: int,
    db: int,
    b: int,
) -> float:
    """Single-generator segment values from Dobrautz et al. Table~1 (tab:single-mateles).

    Parameters
    ----------
    q : str
        Segment label, one of ``"W"``, ``"uR"``, ``"R"``, ``"oR"``,
        ``"uL"``, ``"L"``, ``"oL"``.
    dprime : int
        Bra step value at this segment (0=E, 1=U, 2=L, 3=D).
    d : int
        Ket step value at this segment.
    db : int
        Spin difference b_ket - b_bra at the child node.
    b : int
        Spin quantum number (2S) at the child node of the ket path.

    Returns
    -------
    float
        The segment value W(q; d', d; db, b).
    """

    qcode = _Q_FROM_STR.get(q)
    if qcode is None:
        raise ValueError(f"unknown segment type {q!r}")
    return _segment_value_int(qcode, int(dprime), int(d), int(db), int(b))
