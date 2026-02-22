from __future__ import annotations

import functools
import math
from typing import Final

_TWOS_TOL: Final[float] = 1e-12


def _to_twos(x: float, *, name: str) -> int:
    xf = float(x)
    if not math.isfinite(xf):
        raise ValueError(f"{name} must be finite, got {x!r}")
    t = int(round(2.0 * xf))
    if abs(2.0 * xf - float(t)) > _TWOS_TOL:
        raise ValueError(f"{name} must be integer/half-integer, got {x!r}")
    return t


@functools.lru_cache(maxsize=4096)
def _logfac(n: int) -> float:
    n = int(n)
    if n < 0:
        raise ValueError("factorial arg must be >= 0")
    return math.lgamma(n + 1)


def _tri_ok_twos(tj1: int, tj2: int, tj3: int) -> bool:
    tj1 = int(tj1)
    tj2 = int(tj2)
    tj3 = int(tj3)
    if tj1 < 0 or tj2 < 0 or tj3 < 0:
        return False
    if (tj1 + tj2 + tj3) & 1:
        return False
    if tj3 < abs(tj1 - tj2) or tj3 > tj1 + tj2:
        return False
    return True


def tri_delta_twos(tj1: int, tj2: int, tj3: int) -> float:
    """Triangle coefficient Δ(j1,j2,j3) with doubled-integer inputs."""

    tj1 = int(tj1)
    tj2 = int(tj2)
    tj3 = int(tj3)
    if not _tri_ok_twos(tj1, tj2, tj3):
        return 0.0
    a = (tj1 + tj2 - tj3) // 2
    b = (tj1 - tj2 + tj3) // 2
    c = (-tj1 + tj2 + tj3) // 2
    d = (tj1 + tj2 + tj3) // 2 + 1
    return math.exp(0.5 * (_logfac(a) + _logfac(b) + _logfac(c) - _logfac(d)))


def tri_delta(j1: float, j2: float, j3: float) -> float:
    """Triangle coefficient Δ(j1,j2,j3) for integer/half-integer inputs."""

    return tri_delta_twos(_to_twos(j1, name="j1"), _to_twos(j2, name="j2"), _to_twos(j3, name="j3"))


def _phase_from_twos(exp_twos: int) -> float:
    exp_twos = int(exp_twos)
    if exp_twos & 1:
        raise ValueError("phase exponent must correspond to integer power (even twos)")
    exp_int = exp_twos // 2
    return -1.0 if (exp_int & 1) else 1.0


@functools.lru_cache(maxsize=200_000)
def wigner_3j_twos(tj1: int, tj2: int, tj3: int, tm1: int, tm2: int, tm3: int) -> float:
    """Wigner 3j symbol with doubled-integer inputs (Condon–Shortley)."""

    tj1 = int(tj1)
    tj2 = int(tj2)
    tj3 = int(tj3)
    tm1 = int(tm1)
    tm2 = int(tm2)
    tm3 = int(tm3)

    if (tm1 + tm2 + tm3) != 0:
        return 0.0
    if not _tri_ok_twos(tj1, tj2, tj3):
        return 0.0
    if abs(tm1) > tj1 or abs(tm2) > tj2 or abs(tm3) > tj3:
        return 0.0

    # Parity: j and m must have the same half-integer parity.
    if ((tj1 + tm1) & 1) or ((tj2 + tm2) & 1) or ((tj3 + tm3) & 1):
        return 0.0

    # Common factorial arguments (all integers if the parity checks above pass).
    j1pj2mj3 = (tj1 + tj2 - tj3) // 2
    j1mm1 = (tj1 - tm1) // 2
    j2pm2 = (tj2 + tm2) // 2
    j3mj2pm1 = (tj3 - tj2 + tm1) // 2
    j3mj1mm2 = (tj3 - tj1 - tm2) // 2

    z_min = max(0, -j3mj2pm1, -j3mj1mm2)
    z_max = min(j1pj2mj3, j1mm1, j2pm2)
    if z_min > z_max:
        return 0.0

    log_norm = 0.5 * (
        _logfac((tj1 + tm1) // 2)
        + _logfac((tj1 - tm1) // 2)
        + _logfac((tj2 + tm2) // 2)
        + _logfac((tj2 - tm2) // 2)
        + _logfac((tj3 + tm3) // 2)
        + _logfac((tj3 - tm3) // 2)
    )
    pref = tri_delta_twos(tj1, tj2, tj3) * math.exp(log_norm)
    if pref == 0.0:
        return 0.0

    # (-1)^(j1 - j2 - m3)
    phase = _phase_from_twos(tj1 - tj2 - tm3)

    s = 0.0
    for z in range(int(z_min), int(z_max) + 1):
        den = (
            _logfac(z)
            + _logfac(j1pj2mj3 - z)
            + _logfac(j1mm1 - z)
            + _logfac(j2pm2 - z)
            + _logfac(j3mj2pm1 + z)
            + _logfac(j3mj1mm2 + z)
        )
        term = math.exp(-den)
        if z & 1:
            s -= term
        else:
            s += term

    return float(phase * pref * s)


def wigner_3j(j1: float, j2: float, j3: float, m1: float, m2: float, m3: float) -> float:
    """Wigner 3j symbol for integer/half-integer inputs (Condon–Shortley)."""

    return wigner_3j_twos(
        _to_twos(j1, name="j1"),
        _to_twos(j2, name="j2"),
        _to_twos(j3, name="j3"),
        _to_twos(m1, name="m1"),
        _to_twos(m2, name="m2"),
        _to_twos(m3, name="m3"),
    )


@functools.lru_cache(maxsize=200_000)
def wigner_6j_twos(ta: int, tb: int, tc: int, td: int, te: int, tf: int) -> float:
    """Wigner 6j symbol with doubled-integer inputs (Racah formula)."""

    ta = int(ta)
    tb = int(tb)
    tc = int(tc)
    td = int(td)
    te = int(te)
    tf = int(tf)

    if not (_tri_ok_twos(ta, tb, tc) and _tri_ok_twos(ta, te, tf) and _tri_ok_twos(td, tb, tf) and _tri_ok_twos(td, te, tc)):
        return 0.0

    dprod = (
        tri_delta_twos(ta, tb, tc)
        * tri_delta_twos(ta, te, tf)
        * tri_delta_twos(td, tb, tf)
        * tri_delta_twos(td, te, tc)
    )
    if dprod == 0.0:
        return 0.0

    x1 = (ta + tb + tc) // 2
    x2 = (ta + te + tf) // 2
    x3 = (td + tb + tf) // 2
    x4 = (td + te + tc) // 2

    y1 = (ta + tb + td + te) // 2
    y2 = (ta + tc + td + tf) // 2
    y3 = (tb + tc + te + tf) // 2

    z_min = max(x1, x2, x3, x4)
    z_max = min(y1, y2, y3)
    if z_min > z_max:
        return 0.0

    s = 0.0
    for z in range(int(z_min), int(z_max) + 1):
        den = (
            _logfac(z - x1)
            + _logfac(z - x2)
            + _logfac(z - x3)
            + _logfac(z - x4)
            + _logfac(y1 - z)
            + _logfac(y2 - z)
            + _logfac(y3 - z)
        )
        term = math.exp(_logfac(z + 1) - den)
        if z & 1:
            s -= term
        else:
            s += term

    return float(dprod * s)


def wigner_6j(a: float, b: float, c: float, d: float, e: float, f: float) -> float:
    """Wigner 6j symbol for integer/half-integer inputs (Racah formula)."""

    return wigner_6j_twos(
        _to_twos(a, name="a"),
        _to_twos(b, name="b"),
        _to_twos(c, name="c"),
        _to_twos(d, name="d"),
        _to_twos(e, name="e"),
        _to_twos(f, name="f"),
    )
