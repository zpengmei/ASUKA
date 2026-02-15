from __future__ import annotations

"""Packing basis data into cuERI `BasisCartSoA`.

Input
-----
This module supports two common sources of basis data:
- Basis Set Exchange JSON (see `asuka.frontend.basis_bse`)
- Explicit basis dicts:
    {"H": [[0, [exp, c1, c2, ...], ...], ...], ...}

Output
------
cuERI packed basis objects, which can be fed into cuERI DF / active-space DF
builders without any external package involvement.
"""

import re
from typing import Any

import numpy as np

from asuka.integrals.gto_cart import ncart, primitive_norm_cart_like_pyscf


def _parse_pyscf_shell_entry(entry: Any) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Parse one basis shell entry into (l, exps, coefs) list.

    Supports:
    - [l, [exp, c1, c2, ...], [exp, ...], ...]
    - [l1, l2, ..., [exp, c(l1,ctr1..), c(l2,ctr1..), ...], ...]  (SP shells)
    """

    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        raise ValueError(f"invalid basis shell entry: {entry!r}")

    # Collect angular momenta until the first primitive line.
    ls: list[int] = []
    prim_start = None
    for i, item in enumerate(entry):
        if isinstance(item, (int, np.integer)):
            ls.append(int(item))
            continue
        prim_start = i
        break
    if prim_start is None or not ls:
        raise ValueError(f"invalid basis shell entry header: {entry!r}")

    prim_lines = entry[prim_start:]
    exps: list[float] = []
    coeff_rows: list[list[float]] = []
    for line in prim_lines:
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            raise ValueError(f"invalid primitive line: {line!r}")
        exps.append(float(line[0]))
        coeff_rows.append([float(x) for x in line[1:]])

    exps_arr = np.asarray(exps, dtype=np.float64)
    nprim = int(exps_arr.size)

    coeff_mat = np.asarray(coeff_rows, dtype=np.float64)
    if coeff_mat.ndim != 2 or int(coeff_mat.shape[0]) != nprim:
        raise ValueError("unexpected coefficient matrix shape")

    nL = int(len(ls))
    ncols = int(coeff_mat.shape[1])
    if ncols % nL != 0:
        raise ValueError(f"coeff column count ({ncols}) not divisible by nL ({nL}) for entry {entry!r}")
    nctr = ncols // nL

    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    for i, l in enumerate(ls):
        block = coeff_mat[:, i * nctr : (i + 1) * nctr]
        out.append((int(l), exps_arr, block))
    return out


def parse_pyscf_basis_dict(basis: Any, *, elements: list[str]) -> dict[str, list[tuple[int, np.ndarray, np.ndarray]]]:
    """Parse a basis dict into per-element (l, exps, coefs) shells."""

    if not isinstance(basis, dict):
        raise TypeError("basis must be a dict mapping element symbol -> shell list")
    elements = [str(e).strip() for e in elements]
    if not elements:
        raise ValueError("elements must be non-empty")

    # Be permissive with element-key variants often seen in upstream sources
    # (e.g. lowercase symbols, or labels such as "N1" from Molden pipelines).
    norm_basis: dict[str, Any] = {}
    for key, val in basis.items():
        key_s = str(key).strip()
        m = re.match(r"^([A-Za-z]{1,2})", key_s)
        key_norm = (m.group(1) if m is not None else key_s).capitalize()
        if key_norm not in norm_basis:
            norm_basis[key_norm] = val

    out: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for sym in elements:
        spec = basis.get(sym)
        if spec is None:
            spec = norm_basis.get(sym)
        if spec is None:
            raise KeyError(f"missing basis for element {sym!r}")
        if not isinstance(spec, (list, tuple)):
            raise TypeError(f"basis[{sym!r}] must be a list of shells")
        shells: list[tuple[int, np.ndarray, np.ndarray]] = []
        for entry in spec:
            shells.extend(_parse_pyscf_shell_entry(entry))
        out[sym] = shells
    return out


def pack_cart_basis(
    atoms_bohr: list[tuple[str, np.ndarray]] | tuple[tuple[str, np.ndarray], ...],
    basis_shells: dict[str, list[tuple[int, np.ndarray, np.ndarray]]],
    *,
    expand_contractions: bool = True,
):
    """Pack per-element basis shells into cuERI `BasisCartSoA`."""

    from asuka.cueri.basis_cart import BasisCartSoA  # noqa: PLC0415

    shell_cxyz: list[np.ndarray] = []
    shell_prim_start: list[int] = []
    shell_nprim: list[int] = []
    shell_l: list[int] = []
    shell_ao_start: list[int] = []
    prim_exp: list[float] = []
    prim_coef: list[float] = []

    ao_cursor = 0

    for sym, xyz in atoms_bohr:
        sym = str(sym).strip()
        xyz = np.asarray(xyz, dtype=np.float64).reshape((3,))
        shells = basis_shells.get(sym)
        if shells is None:
            raise KeyError(f"missing basis shells for element {sym!r}")
        for l, exps, coefs in shells:
            l = int(l)
            exps = np.asarray(exps, dtype=np.float64).ravel()
            coefs = np.asarray(coefs, dtype=np.float64)
            if exps.ndim != 1:
                raise ValueError("exps must be 1D")
            if coefs.ndim != 2 or int(coefs.shape[0]) != int(exps.size):
                raise ValueError("coefs must have shape (nprim, nctr)")

            nprim = int(exps.size)
            nctr = int(coefs.shape[1])
            if nprim <= 0 or nctr <= 0:
                raise ValueError("invalid shell sizes")

            norm = primitive_norm_cart_like_pyscf(l, exps)
            if np.any(~np.isfinite(norm)):
                raise ValueError("non-finite primitive normalization")

            ctr_iter = range(nctr) if bool(expand_contractions) else range(1)
            for ctr_id in ctr_iter:
                shell_cxyz.append(xyz)
                shell_prim_start.append(len(prim_exp))
                shell_nprim.append(nprim)
                shell_l.append(l)
                shell_ao_start.append(int(ao_cursor))

                col = coefs[:, int(ctr_id)]
                prim_exp.extend(exps.tolist())
                prim_coef.extend((col * norm).tolist())

                ao_cursor += int(ncart(l))

    return BasisCartSoA(
        shell_cxyz=np.asarray(shell_cxyz, dtype=np.float64),
        shell_prim_start=np.asarray(shell_prim_start, dtype=np.int32),
        shell_nprim=np.asarray(shell_nprim, dtype=np.int32),
        shell_l=np.asarray(shell_l, dtype=np.int32),
        shell_ao_start=np.asarray(shell_ao_start, dtype=np.int32),
        prim_exp=np.asarray(prim_exp, dtype=np.float64),
        prim_coef=np.asarray(prim_coef, dtype=np.float64),
    )


__all__ = ["pack_cart_basis", "parse_pyscf_basis_dict"]
