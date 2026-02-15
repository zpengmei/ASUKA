from __future__ import annotations

from functools import lru_cache


def ncart(l: int) -> int:
    """Calculate the number of Cartesian components for a given angular momentum `l`.

    The number of Cartesian functions for angular momentum `l` corresponds to the
    number of unique `(lx, ly, lz)` tuples such that `lx + ly + lz = l`.
    Formula: `(l + 1) * (l + 2) / 2`.

    Parameters
    ----------
    l : int
        The angular momentum quantum number (l >= 0).

    Returns
    -------
    int
        The number of Cartesian components.
    """
    if l < 0:
        raise ValueError("l must be >= 0")
    return (l + 1) * (l + 2) // 2


@lru_cache(maxsize=None)
def cartesian_components(l: int) -> tuple[tuple[int, int, int], ...]:
    """Generate the sequence of Cartesian exponent tuples `(lx, ly, lz)` for angular momentum `l`.

    The components are ordered according to the standard convention used in PySCF and Libint:
    decreasing `lx`, then decreasing `ly`.
    Example for l=1 (p): `(1,0,0), (0,1,0), (0,0,1)` corresponding to x, y, z.
    Example for l=2 (d): `(2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)` corresponding to xx, xy, xz, yy, yz, zz.

    Parameters
    ----------
    l : int
        The angular momentum quantum number (l >= 0).

    Returns
    -------
    tuple[tuple[int, int, int], ...]
        A tuple of components, where each component is a tuple `(lx, ly, lz)`.
    """
    if l < 0:
        raise ValueError("l must be >= 0")
    out: list[tuple[int, int, int]] = []
    for lx in range(l, -1, -1):
        for ly in range(l - lx, -1, -1):
            lz = l - lx - ly
            out.append((lx, ly, lz))
    return tuple(out)


@lru_cache(maxsize=None)
def _cart_index_map(l: int) -> dict[tuple[int, int, int], int]:
    return {c: i for i, c in enumerate(cartesian_components(l))}


def cart_index(lx: int, ly: int, lz: int) -> int:
    """Return the linear index of a specific Cartesian component `(lx, ly, lz)`.

    This function maps the component `(lx, ly, lz)` to its 0-based index within the
    lexicographically ordered set of components for `l = lx + ly + lz`.

    Parameters
    ----------
    lx, ly, lz : int
        The Cartesian exponents. Must be non-negative.

    Returns
    -------
    int
        The index of the component.
    """
    if lx < 0 or ly < 0 or lz < 0:
        raise ValueError("lx/ly/lz must be >= 0")
    l = lx + ly + lz
    try:
        return _cart_index_map(l)[(lx, ly, lz)]
    except KeyError as exc:  # pragma: no cover (should be unreachable)
        raise ValueError("invalid (lx,ly,lz) for computed l") from exc


def cart_comp_str(lx: int, ly: int, lz: int) -> str:
    """Generate a string representation for a Cartesian component.

    Example: `(1, 1, 0)` becomes "xy".

    Parameters
    ----------
    lx, ly, lz : int
        The Cartesian exponents. Must be non-negative.

    Returns
    -------
    str
        The string representation (e.g., "xx", "xyz").
    """
    if lx < 0 or ly < 0 or lz < 0:
        raise ValueError("lx/ly/lz must be >= 0")
    return ("x" * lx) + ("y" * ly) + ("z" * lz)


__all__ = ["cart_comp_str", "cart_index", "cartesian_components", "ncart"]

