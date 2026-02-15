from __future__ import annotations

import math


def boys_f0(T: float) -> float:
    """Evaluate the Boys function F_0(T).

    The Boys function F_0(T) is defined as the integral from 0 to 1 of exp(-T * t^2) dt.
    Where F_0(T) = err(sqrt(T)) * sqrt(pi) / (2 * sqrt(T)) for T > 0.

    Parameters
    ----------
    T : float
        The argument of the Boys function. Must be non-negative.

    Returns
    -------
    float
        The value of F_0(T).
    """

    if T < 0.0:
        raise ValueError("T must be >= 0")
    if T < 1e-12:
        # series: 1 - T/3 + T^2/10 - ...
        return 1.0 - (T / 3.0) + (T * T / 10.0)
    return 0.5 * math.sqrt(math.pi / T) * math.erf(math.sqrt(T))


def boys_fm(T: float, m: int) -> float:
    """Evaluate the Boys function F_m(T) for a specific order m.

    This function returns a single value F_m(T).

    Parameters
    ----------
    T : float
        The argument of the Boys function. Must be non-negative.
    m : int
        The order of the Boys function. Must be non-negative.

    Returns
    -------
    float
        The value of F_m(T).
    """

    if m < 0:
        raise ValueError("m must be >= 0")
    if T < 0.0:
        raise ValueError("T must be >= 0")
    return boys_fm_list(T, m)[m]


def boys_fm_list(T: float, m_max: int) -> list[float]:
    """Evaluate a sequence of Boys functions [F_0(T), ..., F_{m_max}(T)].

    Computes the Boys functions for all orders up to `m_max` using stable recursion relations.
    Both upward and downward recursions are employed depending on the value of T to maintain precision.

    Parameters
    ----------
    T : float
        The argument of the Boys function. Must be non-negative.
    m_max : int
        The maximum order to compute. Must be non-negative.

    Returns
    -------
    list[float]
        A list of length `m_max + 1` containing the values [F_0(T), ..., F_{m_max}(T)].
    """

    if m_max < 0:
        raise ValueError("m_max must be >= 0")
    if T < 0.0:
        raise ValueError("T must be >= 0")
    if m_max == 0:
        return [boys_f0(T)]

    # For moderate T, upward recursion can lose many digits for large m due to cancellation
    # against exp(-T). Use a series top-evaluation + downward recursion in that regime.
    if T < 5.0:
        # Compute F_{m_max} from series, then recurse downward:
        #   F_{m-1} = (2T F_m + exp(-T)) / (2m-1)
        out = [0.0] * (m_max + 1)
        term = 1.0
        Fm = 0.0
        for k in range(120):
            Fm += term / (2 * m_max + 2 * k + 1)
            term *= -T / float(k + 1)
        out[m_max] = Fm
        e = math.exp(-T)
        for m in range(m_max, 0, -1):
            out[m - 1] = (2.0 * T * out[m] + e) / float(2 * m - 1)
        return out

    out = [0.0] * (m_max + 1)
    out[0] = boys_f0(T)
    e = math.exp(-T)
    for m in range(1, m_max + 1):
        out[m] = ((2 * m - 1) * out[m - 1] - e) / (2 * T)
    return out


__all__ = ["boys_f0", "boys_fm", "boys_fm_list"]
