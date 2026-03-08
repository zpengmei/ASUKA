"""Superindex ordering bridge: ASUKA C-order to Molcas Fortran-order."""
from __future__ import annotations

import os

import numpy as np


def _superindex_c_to_f_perm(case: int, nash: int) -> np.ndarray | None:
    """Compute permutation from ASUKA C-order to Molcas F-order superindices.

    ASUKA builds superindex maps in C (row-major) order while the Molcas
    translation code (clagdx_cases.py, olagns.py) iterates in Fortran
    (column-major) order. This only matters for:

    * Cases 1 (A), 4 (C): triple indices (t,u,v)
      ASUKA: t*nash²+u*nash+v  vs  Molcas: t+nash*u+nash²*v
    * Case 5 (D): full pair indices in 2-block structure
      ASUKA: t*nash+u  vs  Molcas: t+nash*u

    Symmetric/antisymmetric pairs and single-index cases have
    identical ordering in both conventions.

    Returns
    -------
    perm : ndarray or None
        ``perm[molcas_idx] = asuka_idx``. Apply as
        ``smat_molcas = smat_asuka[np.ix_(perm, perm)]``.
        *None* if no permutation is needed.
    """
    if nash <= 1:
        return None

    if case in (1, 4):  # triples
        perm = np.arange(nash**3).reshape(nash, nash, nash).ravel(order="F")
        if np.array_equal(perm, np.arange(nash**3)):
            return None
        return perm

    if case == 5:  # full pairs, 2-block
        # Debug switch: disable Case-5 permutation to isolate D-block
        # superindex mapping effects in parity diagnostics.
        # Default remains "on" (Molcas Fortran-order permutation enabled).
        _mode = str(os.environ.get("ASUKA_PT2LAG_CASE5_PERM", "on")).strip().lower()
        if _mode in {"0", "off", "false", "no"}:
            return None
        ntu = nash * nash
        perm_pairs = np.arange(ntu).reshape(nash, nash).ravel(order="F")
        perm = np.concatenate([perm_pairs, perm_pairs + ntu])
        if np.array_equal(perm, np.arange(2 * ntu)):
            return None
        return perm

    return None


def _pair_sym_c_to_f_perm(n: int) -> np.ndarray | None:
    """Permutation from ASUKA C-order to Molcas order for p>=q pairs.

    Returns ``perm[molcas_idx] = asuka_idx``.
    """
    if n <= 1:
        return None
    asuka_pairs = [(p, q) for p in range(n) for q in range(p + 1)]
    asuka_pos = {pq: idx for idx, pq in enumerate(asuka_pairs)}
    molcas_pairs = [(p, q) for q in range(n) for p in range(q, n)]
    perm = np.asarray([asuka_pos[pq] for pq in molcas_pairs], dtype=np.int64)
    if np.array_equal(perm, np.arange(perm.size, dtype=np.int64)):
        return None
    return perm


def _pair_asym_c_to_f_perm(n: int) -> np.ndarray | None:
    """Permutation from ASUKA C-order to Molcas order for p>q pairs.

    Returns ``perm[molcas_idx] = asuka_idx``.
    """
    if n <= 1:
        return None
    asuka_pairs = [(p, q) for p in range(n) for q in range(p)]
    asuka_pos = {pq: idx for idx, pq in enumerate(asuka_pairs)}
    molcas_pairs = [(p, q) for q in range(n) for p in range(q + 1, n)]
    perm = np.asarray([asuka_pos[pq] for pq in molcas_pairs], dtype=np.int64)
    if np.array_equal(perm, np.arange(perm.size, dtype=np.int64)):
        return None
    return perm


def _expand_block_perm(base_perm: np.ndarray | None, block: int) -> np.ndarray | None:
    """Expand a base-index permutation to packed block-major linear indices.

    Input layout is ``idx = inner + block * outer`` (inner-fast).
    """
    if base_perm is None:
        return None
    if int(block) <= 0:
        raise ValueError(f"invalid block size: {block}")
    b = int(base_perm.size)
    out = np.empty((b * int(block),), dtype=np.int64)
    k = 0
    for om in range(b):
        oa = int(base_perm[om])
        start = oa * int(block)
        for inner in range(int(block)):
            out[k] = start + inner
            k += 1
    if np.array_equal(out, np.arange(out.size, dtype=np.int64)):
        return None
    return out


def _case_c_to_f_perms(case: int, nish: int, nash: int, nssh: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return active/external permutations (Molcas index -> ASUKA index).

    Active permutation is applied on rows (active superindex). External
    permutation is applied on columns (inactive/virtual superindex).
    """
    p_act: np.ndarray | None = None
    p_ext: np.ndarray | None = None

    # Active superindex mapping.
    if case in (1, 4, 5):
        p_act = _superindex_c_to_f_perm(case, nash)
    elif case in (2, 8):
        p_act = _pair_sym_c_to_f_perm(nash)
    elif case in (3, 9):
        p_act = _pair_asym_c_to_f_perm(nash)
    # External superindex mapping.
    if case == 2:  # B+: inactive symmetric pair
        p_ext = _pair_sym_c_to_f_perm(nish)
    elif case == 3:  # B-: inactive antisymmetric pair
        p_ext = _pair_asym_c_to_f_perm(nish)
    elif case == 6:  # E+: (a, igej), inner-fast a
        p_ext = _expand_block_perm(_pair_sym_c_to_f_perm(nish), nssh)
    elif case == 7:  # E-: (a, igtj), inner-fast a
        p_ext = _expand_block_perm(_pair_asym_c_to_f_perm(nish), nssh)
    elif case == 8:  # F+: virtual symmetric pair
        p_ext = _pair_sym_c_to_f_perm(nssh)
    elif case == 9:  # F-: virtual antisymmetric pair
        p_ext = _pair_asym_c_to_f_perm(nssh)
    elif case == 10:  # G+: (i, ageb), inner-fast i
        p_ext = _expand_block_perm(_pair_sym_c_to_f_perm(nssh), nish)
    elif case == 11:  # G-: (i, agtb), inner-fast i
        p_ext = _expand_block_perm(_pair_asym_c_to_f_perm(nssh), nish)
    elif case == 12:  # H+: inactive symmetric pair
        p_ext = _pair_sym_c_to_f_perm(nish)
    elif case == 13:  # H-: inactive antisymmetric pair
        p_ext = _pair_asym_c_to_f_perm(nish)

    return p_act, p_ext


def _parse_pair_perm_cases_env(raw: str) -> set[int] | None:
    """Parse optional case filter for pair-permutation debug mode.

    ``raw`` accepts comma/space/semicolon separated positive integers.
    Returns ``None`` when no valid case ids are provided (no filter).
    """
    s = str(raw).strip()
    if not s:
        return None
    toks = (
        s.replace(";", ",")
        .replace(":", ",")
        .replace("|", ",")
        .replace("\t", ",")
        .replace(" ", ",")
        .split(",")
    )
    out: set[int] = set()
    for t in toks:
        tt = str(t).strip()
        if not tt:
            continue
        try:
            iv = int(tt)
        except Exception:
            continue
        if iv > 0:
            out.add(int(iv))
    return out if out else None


def _select_case_pair_perms(
    *,
    case: int,
    perm_act_raw: np.ndarray | None,
    perm_ext_raw: np.ndarray | None,
    pair_perm_mode: str,
    pair_perm_cases: set[int] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Apply pair-permutation policy for a given case.

    Behavior matches legacy defaults:
    - ``pair_perm_mode=off`` keeps only A/C/D (cases 1/4/5) remaps.
    - ``pair_perm_mode=on`` enables pair-case remaps for all cases.
    - ``pair_perm_mode=ext`` enables only external remaps for non-A/C/D cases.

    Optional ``pair_perm_cases`` filter applies only to non-A/C/D cases.
    """
    c = int(case)
    mode = str(pair_perm_mode).strip().lower()
    enabled = mode not in {"", "0", "off", "false", "no"}
    ext_only = mode in {"ext", "external"}

    perm_act = perm_act_raw
    perm_ext = perm_ext_raw

    if not enabled:
        if c not in (1, 4, 5):
            perm_act = None
            perm_ext = None
    else:
        # Restrict pair-case remaps to a selected subset when requested.
        if c not in (1, 4, 5) and pair_perm_cases is not None and c not in pair_perm_cases:
            perm_act = None
            perm_ext = None
        elif ext_only and c not in (1, 4, 5):
            # Debug mode: keep pair external-index remap, but avoid active
            # pair re-diagonalization changes.
            perm_act = None

    return perm_act, perm_ext
