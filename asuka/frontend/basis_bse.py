from __future__ import annotations

"""Basis-set loading via Basis Set Exchange (optional dependency).
"""

import json
from typing import Any

import numpy as np

from .periodic_table import atomic_number


def _require_bse():
    try:
        import basis_set_exchange as bse  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "basis_set_exchange is required to load basis sets by name. "
            "Install it (e.g. `pip install basis_set_exchange`) or pass explicit basis data."
        ) from e
    return bse


def _parse_bse_shell(shell: dict[str, Any]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    ams = [int(x) for x in shell["angular_momentum"]]
    if not ams:
        raise ValueError("invalid BSE shell: empty angular_momentum")

    exps = np.asarray(shell["exponents"], dtype=np.float64)
    if exps.ndim != 1 or exps.size == 0:
        raise ValueError("invalid BSE shell: empty exponents")
    nprim = int(exps.size)

    coeff = shell["coefficients"]
    nL = int(len(ams))

    # BSE schema uses a slightly shape-dependent encoding:
    # - if nL==1: coefficients are a list of contractions, each a length-nprim list
    # - if nL>1 and nctr==1: coefficients may be a list of length nL (no contraction axis)
    # - if nL>1 and nctr>1: coefficients is list of contractions, each a list of length nL
    if nL == 1:
        coeff_arr = np.asarray(coeff, dtype=np.float64)
        if coeff_arr.ndim != 2 or int(coeff_arr.shape[1]) != nprim:
            raise ValueError("unexpected BSE coefficients shape for nL==1 shell")
        coef = coeff_arr.T  # (nprim, nctr)
        return [(ams[0], exps, coef)]

    # nL > 1
    # Heuristic: if coeff is length nL and each entry is a length-nprim list of scalars,
    # treat it as the nctr==1 fast form (no contraction axis).
    if isinstance(coeff, list) and len(coeff) == nL and all(isinstance(x, list) and x and not isinstance(x[0], list) for x in coeff):
        out: list[tuple[int, np.ndarray, np.ndarray]] = []
        for l, vec in zip(ams, coeff, strict=True):
            v = np.asarray(vec, dtype=np.float64)
            if v.shape != (nprim,):
                raise ValueError("unexpected BSE coefficient vector shape for multi-l shell")
            out.append((int(l), exps, v.reshape((nprim, 1))))
        return out

    # General form: list of contractions, each with per-l coefficient vectors.
    coeff_arr = np.asarray(coeff, dtype=np.float64)
    if coeff_arr.ndim != 3 or int(coeff_arr.shape[1]) != nL or int(coeff_arr.shape[2]) != nprim:
        raise ValueError("unexpected BSE coefficients shape for multi-l shell")
    # coeff_arr: (nctr, nL, nprim)
    out2: list[tuple[int, np.ndarray, np.ndarray]] = []
    for i, l in enumerate(ams):
        coef = coeff_arr[:, i, :].T  # (nprim, nctr)
        out2.append((int(l), exps, coef))
    return out2


def load_element_basis_shells(basis_name: str, *, element: str) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Load basis shells for one element using BSE (as (l, exps, coefs))."""

    bse = _require_bse()
    sym = str(element).strip()
    Z = atomic_number(sym)
    s = bse.get_basis(str(basis_name), elements=[sym], fmt="json", header=False)
    data = json.loads(s)
    elt = data["elements"][str(Z)]
    shells = elt["electron_shells"]
    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    for sh in shells:
        out.extend(_parse_bse_shell(sh))
    return out


def load_basis_shells(basis_name: str, *, elements: list[str]) -> dict[str, list[tuple[int, np.ndarray, np.ndarray]]]:
    """Load per-element basis shells using BSE (as (l, exps, coefs))."""

    bse = _require_bse()
    elements = [str(e).strip() for e in elements]
    if not elements:
        raise ValueError("elements must be non-empty")

    s = bse.get_basis(str(basis_name), elements=elements, fmt="json", header=False)
    data = json.loads(s)
    out: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for sym in elements:
        Z = atomic_number(sym)
        shells = data["elements"][str(Z)]["electron_shells"]
        buf: list[tuple[int, np.ndarray, np.ndarray]] = []
        for sh in shells:
            buf.extend(_parse_bse_shell(sh))
        out[sym] = buf
    return out


def load_autoaux_shells(
    orbital_basis_name: str,
    *,
    elements: list[str],
) -> tuple[str, dict[str, list[tuple[int, np.ndarray, np.ndarray]]]]:
    """Load the BSE autoaux auxiliary basis corresponding to an orbital basis."""

    bse = _require_bse()
    elements = [str(e).strip() for e in elements]
    if not elements:
        raise ValueError("elements must be non-empty")

    s = bse.get_basis(str(orbital_basis_name), elements=elements, fmt="json", header=False, get_aux=1)
    data = json.loads(s)
    aux_name = str(data.get("name", ""))
    if not aux_name:
        raise ValueError("BSE did not return an auxiliary basis name")

    out: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for sym in elements:
        Z = atomic_number(sym)
        shells = data["elements"][str(Z)]["electron_shells"]
        buf: list[tuple[int, np.ndarray, np.ndarray]] = []
        for sh in shells:
            buf.extend(_parse_bse_shell(sh))
        out[sym] = buf
    return aux_name, out


__all__ = ["load_autoaux_shells", "load_basis_shells", "load_element_basis_shells"]

