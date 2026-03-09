from __future__ import annotations

"""Gaussian .gbs format parser for bundled basis sets (e.g. Minnesota ma-XZVP).

This module parses Gaussian-format `.gbs` files shipped in ``basis_data/``
and converts them into the ASUKA internal representation used by
:func:`basis_bse.load_basis_shells`.
"""

import os
import re
from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------------------
# Angular momentum label → quantum number
# ---------------------------------------------------------------------------

_AM_MAP: dict[str, int] = {
    "S": 0,
    "P": 1,
    "D": 2,
    "F": 3,
    "G": 4,
    "H": 5,
    "I": 6,
}

# ---------------------------------------------------------------------------
# Bundled basis-set name → filename mapping (case-insensitive lookup)
# ---------------------------------------------------------------------------

_BUNDLED_BASES: dict[str, str] = {
    "ma-svp": "ma-SVP.gbs",
    "ma-svpp": "ma-SVPP.gbs",
    "ma-tzvp": "ma-TZVP.gbs",
    "ma-tzvpp": "ma-TZVPP.gbs",
    "ma-qzvp": "ma-QZVP.gbs",
    "ma-qzvpp": "ma-QZVPP.gbs",
}

# Additional accepted aliases -> canonical key in _BUNDLED_BASES.
_BUNDLED_ALIASES: dict[str, str] = {
    "ma-sv(p)": "ma-svpp",
    "ma-def2-svp": "ma-svp",
    "ma-def2-sv(p)": "ma-svpp",
    "ma-def2-svpp": "ma-svpp",
    "ma-def2-tzvp": "ma-tzvp",
    "ma-def2-tzvpp": "ma-tzvpp",
    "ma-def2-qzvp": "ma-qzvp",
    "ma-def2-qzvpp": "ma-qzvpp",
}

# Autoaux fallback: when BSE cannot generate autoaux for a bundled basis,
# fall back to the corresponding def2 orbital basis for autoaux generation.
_AUTOAUX_FALLBACK: dict[str, str] = {
    "ma-svp": "def2-svp",
    "ma-svpp": "def2-sv(p)",
    "ma-tzvp": "def2-tzvp",
    "ma-tzvpp": "def2-tzvpp",
    "ma-qzvp": "def2-qzvp",
    "ma-qzvpp": "def2-qzvpp",
}

_DATA_DIR = os.path.join(os.path.dirname(__file__), "basis_data")

# Fortran D-exponent regex:  e.g. 0.123D-04  or  1.23D+02
_FORTRAN_D_RE = re.compile(r"(\d)D([+-])", re.IGNORECASE)


def _fortran_float(s: str) -> float:
    """Convert a string that may use Fortran ``D`` exponent notation to float."""
    return float(_FORTRAN_D_RE.sub(r"\1E\2", s))


def _normalize_basis_key(basis_name: str) -> str:
    """Normalize basis name into the canonical bundled-basis key."""
    key = str(basis_name).strip().lower().replace("_", "-")
    key = re.sub(r"\s+", "", key)
    if key.startswith("ma-def2-"):
        key = "ma-" + key[len("ma-def2-") :]
    return _BUNDLED_ALIASES.get(key, key)


# ---------------------------------------------------------------------------
# .gbs parser
# ---------------------------------------------------------------------------


def parse_gbs_file(filepath: str) -> dict[str, list[tuple[int, np.ndarray, np.ndarray]]]:
    """Parse a Gaussian ``.gbs`` file into ASUKA's internal basis format.

    Parameters
    ----------
    filepath : str
        Path to the ``.gbs`` file.

    Returns
    -------
    dict[str, list[tuple[int, ndarray, ndarray]]]
        Mapping from element symbol (e.g. ``"C"``) to a list of shells.
        Each shell is ``(angular_momentum_l, exponents, coefficients)``
        where *exponents* has shape ``(nprim,)`` and *coefficients* has
        shape ``(nprim, nctr)``.
    """

    with open(filepath, encoding="utf-8") as fh:
        raw = fh.read()

    # Split on element-separator lines (****)
    blocks = re.split(r"^\*\*\*\*\s*$", raw, flags=re.MULTILINE)

    result: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}

    for block in blocks:
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue

        # First non-blank line is the element header: ``-Sym   0``
        header = lines[0].strip()
        # Strip leading dash (UMN format) and trailing ``0``
        hdr_parts = header.lstrip("-").split()
        if not hdr_parts:
            raise ValueError(f"invalid element header in {filepath!r}: {header!r}")
        sym = hdr_parts[0]
        # Normalise symbol: first letter upper, rest lower
        sym = sym[0].upper() + sym[1:].lower() if len(sym) > 1 else sym.upper()

        elt_shells: list[tuple[int, np.ndarray, np.ndarray]] = []
        idx = 1  # skip element header line
        while idx < len(lines):
            # Shell header line: ``S  3  1.00``
            parts = lines[idx].split()
            if len(parts) < 3:
                idx += 1
                continue

            shell_label = parts[0].upper()
            if shell_label not in _AM_MAP:
                idx += 1
                continue

            am = _AM_MAP[shell_label]
            nprim = int(parts[1])
            # parts[2] is the scale factor (always 1.00 in practice)
            idx += 1

            exps: list[float] = []
            coefs: list[list[float]] = []
            for _ in range(nprim):
                if idx >= len(lines):
                    break
                vals = lines[idx].split()
                idx += 1
                if len(vals) < 2:
                    raise ValueError(f"invalid primitive row in {filepath!r}: {lines[idx - 1]!r}")
                exps.append(_fortran_float(vals[0]))
                coefs.append([_fortran_float(v) for v in vals[1:]])

            if len(exps) != nprim:
                raise ValueError(
                    f"shell {shell_label} for {sym} in {filepath!r} expected {nprim} primitives, got {len(exps)}"
                )
            nctr = len(coefs[0])
            if nctr == 0 or any(len(row) != nctr for row in coefs):
                raise ValueError(f"inconsistent contraction columns in {filepath!r} for {sym} {shell_label}")

            exp_arr = np.asarray(exps, dtype=np.float64)
            # coefs is (nprim, nctr) – typically nctr==1 for .gbs files
            coef_arr = np.asarray(coefs, dtype=np.float64)
            if coef_arr.ndim == 1:
                coef_arr = coef_arr.reshape(-1, 1)

            elt_shells.append((am, exp_arr, coef_arr))

        if elt_shells:
            if sym in result:
                # Merge (e.g. if an element appears in multiple blocks)
                result[sym].extend(elt_shells)
            else:
                result[sym] = elt_shells

    return result


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_bundled_basis(basis_name: str) -> bool:
    """Return True if *basis_name* is a bundled basis we can load locally."""
    return _normalize_basis_key(basis_name) in _BUNDLED_BASES


@lru_cache(maxsize=None)
def _parse_bundled_by_key(key: str) -> dict[str, list[tuple[int, np.ndarray, np.ndarray]]]:
    if key not in _BUNDLED_BASES:
        raise ValueError(f"Basis set {key!r} is not bundled")
    filepath = os.path.join(_DATA_DIR, _BUNDLED_BASES[key])
    return parse_gbs_file(filepath)


def load_bundled_basis(
    basis_name: str,
    *,
    elements: list[str],
) -> dict[str, list[tuple[int, np.ndarray, np.ndarray]]]:
    """Load a bundled ``.gbs`` basis set for the requested elements.

    Parameters
    ----------
    basis_name : str
        Basis set name (case-insensitive), e.g. ``"ma-TZVP"``.
    elements : list[str]
        Element symbols to load (e.g. ``["H", "C", "O"]``).

    Returns
    -------
    dict[str, list[tuple[int, ndarray, ndarray]]]
        Per-element shell data in ASUKA's internal format.

    Raises
    ------
    ValueError
        If the basis is not bundled or an element is missing from the file.
    """

    key = _normalize_basis_key(basis_name)
    if key not in _BUNDLED_BASES:
        raise ValueError(f"Basis set {basis_name!r} is not bundled")

    all_shells = _parse_bundled_by_key(key)

    out: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for sym in elements:
        sym_clean = sym.strip()
        sym_norm = sym_clean[0].upper() + sym_clean[1:].lower() if len(sym_clean) > 1 else sym_clean.upper()
        if sym_norm not in all_shells:
            avail = ", ".join(sorted(all_shells.keys()))
            raise ValueError(
                f"Element {sym_norm!r} not found in bundled basis {basis_name!r} (available: {avail})"
            )
        # Return defensive copies so callers cannot mutate cached basis data.
        out[sym_clean] = [(l, exps.copy(), coefs.copy()) for l, exps, coefs in all_shells[sym_norm]]
    return out


def autoaux_fallback_name(basis_name: str) -> str | None:
    """Return the def2 basis name to use for autoaux generation, or None."""
    return _AUTOAUX_FALLBACK.get(_normalize_basis_key(basis_name))


__all__ = [
    "autoaux_fallback_name",
    "is_bundled_basis",
    "load_bundled_basis",
    "parse_gbs_file",
]
