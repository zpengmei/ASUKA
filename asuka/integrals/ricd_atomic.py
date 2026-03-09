from __future__ import annotations

"""Atomic-level utilities for the RICD auxiliary-basis generator.

Extracts per-atom AO shell slices from a molecular ``BasisCartSoA``,
groups atoms into unique basis types by fingerprint, and implements
the Molcas-style SHAC/KHAC shell-pair pruning heuristic.
"""

from collections import defaultdict

import numpy as np

from asuka.cueri.cart import ncart
from asuka.frontend.periodic_table import atomic_number as _z_from_symbol
from asuka.integrals.gto_cart import primitive_norm_cart_like_pyscf
from asuka.integrals.ricd_types import AtomicBasisType, RICDShell


# ---------------------------------------------------------------------------
# Atomic-number block values for SHAC pruning (Molcas convention)
# ---------------------------------------------------------------------------

def _ival_from_z(Z: int) -> int:
    """Return the Molcas ``i_val`` block index for atomic number *Z*."""
    Z = int(Z)
    if Z <= 2:
        return 0
    if Z <= 18:
        return 1
    if Z <= 54:
        return 2
    return 3


def molcas_keep_shell_from_atomic_number(Z: int, n_shell: int) -> int:
    """Maximum ``i + j`` allowed for a shell pair under SHAC pruning.

    Parameters
    ----------
    Z : int
        Atomic number.
    n_shell : int
        Number of AO shells on this atom.

    Returns
    -------
    int
        Shell pair (i, j) is retained iff ``i + j <= keep_shell``.
    """
    Z = int(Z)
    n_shell = int(n_shell)
    if n_shell <= 0:
        return 0
    n_test = n_shell - 1
    keep_all = 2 * n_test
    i_val = _ival_from_z(Z)
    i_Z = max(0, n_test - i_val)
    return keep_all - i_Z


# ---------------------------------------------------------------------------
# Atom ↔ shell mapping
# ---------------------------------------------------------------------------

def _assign_shells_to_atoms(
    atoms_bohr: list[tuple[str, np.ndarray]],
    shell_cxyz: np.ndarray,
) -> list[list[int]]:
    """Map each molecular shell to the nearest atom by center coordinate.

    Returns a list of shell-index lists, one per atom.
    """
    n_atoms = len(atoms_bohr)
    n_shells = int(shell_cxyz.shape[0])
    if n_atoms == 0 or n_shells == 0:
        return [[] for _ in range(n_atoms)]

    atom_xyz = np.array([np.asarray(xyz, dtype=np.float64).ravel() for _, xyz in atoms_bohr])

    # Assign each shell to the closest atom (exact match expected after packing)
    shells_per_atom: list[list[int]] = [[] for _ in range(n_atoms)]
    for sh in range(n_shells):
        c = shell_cxyz[sh]
        diff = atom_xyz - c[np.newaxis, :]
        dist_sq = np.sum(diff * diff, axis=1)
        iatom = int(np.argmin(dist_sq))
        shells_per_atom[iatom].append(sh)

    return shells_per_atom


def _basis_type_fingerprint(
    symbol: str,
    shell_l: np.ndarray,
    shell_nprim: np.ndarray,
    prim_exp: np.ndarray,
    prim_coef: np.ndarray,
    shell_prim_start: np.ndarray,
    local_shell_indices: list[int],
    molecular_shell_indices: list[int],
) -> str:
    """Build a deterministic fingerprint for the AO basis on one atom.

    Encodes element symbol, shell angular momenta, and primitive data so
    that atoms with identical AO basis produce the same fingerprint string.
    """
    parts: list[str] = [symbol]
    for loc_idx, mol_idx in zip(local_shell_indices, molecular_shell_indices):
        l = int(shell_l[mol_idx])
        nprim = int(shell_nprim[mol_idx])
        ps = int(shell_prim_start[mol_idx])
        exps = prim_exp[ps : ps + nprim]
        coefs = prim_coef[ps : ps + nprim]
        # Use repr with limited precision to avoid float noise
        exp_str = ",".join(f"{e:.14e}" for e in exps)
        coef_str = ",".join(f"{c:.14e}" for c in coefs)
        parts.append(f"L{l}N{nprim}E({exp_str})C({coef_str})")
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Public extraction API
# ---------------------------------------------------------------------------

def extract_atomic_basis_types(
    ao_basis: "BasisCartSoA",  # noqa: F821
    atoms_bohr: list[tuple[str, np.ndarray]],
) -> list[AtomicBasisType]:
    """Group atoms by identical AO basis type and return one entry per type.

    Parameters
    ----------
    ao_basis : BasisCartSoA
        Molecular AO basis (expanded, Cartesian).
    atoms_bohr : list of (symbol, xyz)
        Atoms with coordinates in Bohr.

    Returns
    -------
    list[AtomicBasisType]
        One entry per unique basis type, ordered by first occurrence.
    """
    shell_cxyz = np.asarray(ao_basis.shell_cxyz, dtype=np.float64)
    shell_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    shell_nprim = np.asarray(ao_basis.shell_nprim, dtype=np.int32).ravel()
    shell_prim_start = np.asarray(ao_basis.shell_prim_start, dtype=np.int32).ravel()
    prim_exp = np.asarray(ao_basis.prim_exp, dtype=np.float64).ravel()
    prim_coef = np.asarray(ao_basis.prim_coef, dtype=np.float64).ravel()

    shells_per_atom = _assign_shells_to_atoms(atoms_bohr, shell_cxyz)

    # Fingerprint each atom and group
    fp_to_atom_indices: dict[str, list[int]] = {}
    fp_order: list[str] = []

    for iatom, mol_shells in enumerate(shells_per_atom):
        local_indices = list(range(len(mol_shells)))
        fp = _basis_type_fingerprint(
            atoms_bohr[iatom][0],
            shell_l, shell_nprim, prim_exp, prim_coef, shell_prim_start,
            local_indices, mol_shells,
        )
        if fp not in fp_to_atom_indices:
            fp_to_atom_indices[fp] = []
            fp_order.append(fp)
        fp_to_atom_indices[fp].append(iatom)

    # Build AtomicBasisType entries
    result: list[AtomicBasisType] = []
    for fp in fp_order:
        atom_indices = fp_to_atom_indices[fp]
        rep_atom = atom_indices[0]
        sym = str(atoms_bohr[rep_atom][0]).strip()
        Z = _z_from_symbol(sym)
        rep_xyz = np.asarray(atoms_bohr[rep_atom][1], dtype=np.float64).reshape(3)
        rep_mol_shells = shells_per_atom[rep_atom]

        local_meta: list[tuple[int, int, int, int]] = []
        for loc_idx, mol_idx in enumerate(rep_mol_shells):
            local_meta.append((
                loc_idx,
                int(shell_l[mol_idx]),
                int(shell_nprim[mol_idx]),
                int(shell_prim_start[mol_idx]),
            ))

        result.append(AtomicBasisType(
            key=fp,
            symbol=sym,
            atomic_number=Z,
            atom_indices=tuple(atom_indices),
            rep_center=rep_xyz,
            local_shell_meta=tuple(local_meta),
        ))

    return result


# ---------------------------------------------------------------------------
# Shell-pair enumeration
# ---------------------------------------------------------------------------

def iter_retained_shell_pairs(
    atom_type: AtomicBasisType,
    *,
    skip_high_ac: bool = False,
) -> list[tuple[int, int]]:
    """Return retained same-center AO shell pairs for RICD candidate generation.

    Parameters
    ----------
    atom_type : AtomicBasisType
        The atomic basis type.
    skip_high_ac : bool
        If *True*, apply the Molcas SHAC pruning.  Default (*False*) keeps
        all pairs (KHAC).

    Returns
    -------
    list of (int, int)
        Local shell-index pairs ``(ia, ib)`` with ``ia >= ib``.
    """
    n_shell = len(atom_type.local_shell_meta)
    if n_shell == 0:
        return []

    if skip_high_ac:
        keep = molcas_keep_shell_from_atomic_number(atom_type.atomic_number, n_shell)
    else:
        keep = 2 * (n_shell - 1)  # keep_all

    pairs: list[tuple[int, int]] = []
    for ia in range(n_shell):
        for ib in range(ia + 1):
            if ia + ib <= keep:
                pairs.append((ia, ib))
    return pairs


# ---------------------------------------------------------------------------
# Shell view helper
# ---------------------------------------------------------------------------

class _ShellView:
    """Lightweight accessor for one shell's primitives inside a BasisCartSoA."""

    __slots__ = ("l", "nprim", "exp", "coef")

    def __init__(self, l: int, nprim: int, exp: np.ndarray, coef: np.ndarray):
        self.l = l
        self.nprim = nprim
        self.exp = exp
        self.coef = coef


def _shell_view_from_basis(
    ao_basis: "BasisCartSoA",  # noqa: F821
    mol_shell_idx: int,
) -> _ShellView:
    """Extract a primitive-data view for one shell."""
    l = int(np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()[mol_shell_idx])
    nprim = int(np.asarray(ao_basis.shell_nprim, dtype=np.int32).ravel()[mol_shell_idx])
    ps = int(np.asarray(ao_basis.shell_prim_start, dtype=np.int32).ravel()[mol_shell_idx])
    exp = np.asarray(ao_basis.prim_exp, dtype=np.float64).ravel()[ps : ps + nprim]
    coef = np.asarray(ao_basis.prim_coef, dtype=np.float64).ravel()[ps : ps + nprim]
    return _ShellView(l, nprim, exp, coef)


def _shell_view_from_meta(
    atom_type: AtomicBasisType,
    ao_basis: "BasisCartSoA",  # noqa: F821
    local_idx: int,
) -> _ShellView:
    """Extract a shell view using AtomicBasisType metadata."""
    _loc_idx, l, nprim, prim_start = atom_type.local_shell_meta[int(local_idx)]
    nprim_i = int(nprim)
    ps = int(prim_start)
    exp = np.asarray(ao_basis.prim_exp, dtype=np.float64).ravel()[ps : ps + nprim_i]
    coef = np.asarray(ao_basis.prim_coef, dtype=np.float64).ravel()[ps : ps + nprim_i]
    return _ShellView(int(l), nprim_i, exp, coef)


# ---------------------------------------------------------------------------
# Contracted candidate shell construction (design doc §7.2.2)
# ---------------------------------------------------------------------------

def _merge_duplicate_exponents(
    exps: list[float],
    coefs: list[float],
    *,
    rtol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge exponent duplicates by summing coefficients.

    Returns (unique_exp, summed_coef) sorted by descending exponent.
    """
    if not exps:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # Sort by exponent descending
    idx = np.argsort(exps)[::-1]
    e_sorted = np.asarray(exps, dtype=np.float64)[idx]
    c_sorted = np.asarray(coefs, dtype=np.float64)[idx]

    merged_e: list[float] = [float(e_sorted[0])]
    merged_c: list[float] = [float(c_sorted[0])]

    for k in range(1, len(e_sorted)):
        e_k = float(e_sorted[k])
        if abs(e_k - merged_e[-1]) <= rtol * max(abs(e_k), abs(merged_e[-1]), 1.0e-30):
            merged_c[-1] += float(c_sorted[k])
        else:
            merged_e.append(e_k)
            merged_c.append(float(c_sorted[k]))

    return np.asarray(merged_e, dtype=np.float64), np.asarray(merged_c, dtype=np.float64)


def _rounded_exp_key(g: float, *, digits: int = 12) -> int:
    """Hash-friendly rounded exponent key for deduplication."""
    return round(float(g) * 10**digits)


def make_contracted_candidate_shells(
    atom_type: AtomicBasisType,
    ao_basis: "BasisCartSoA",  # noqa: F821
    *,
    skip_high_ac: bool = False,
) -> list[RICDShell]:
    """Build the contracted candidate product-shell pool (aCD pool).

    Implements design doc §7.2.2: for each retained shell pair (a, b),
    build one contracted auxiliary shell with L = l_a + l_b.

    Parameters
    ----------
    atom_type : AtomicBasisType
        Atomic basis type.
    ao_basis : BasisCartSoA
        The molecular AO basis (primitives include normalization).
    skip_high_ac : bool
        Apply SHAC pruning if *True*.

    Returns
    -------
    list[RICDShell]
        Contracted candidate shells (one per retained shell pair, deduplicated).
    """
    pairs = iter_retained_shell_pairs(atom_type, skip_high_ac=skip_high_ac)
    shells: list[RICDShell] = []

    for ia, ib in pairs:
        sh_a = _shell_view_from_meta(atom_type, ao_basis, ia)
        sh_b = _shell_view_from_meta(atom_type, ao_basis, ib)

        L = sh_a.l + sh_b.l

        gamma_list: list[float] = []
        coeff_list: list[float] = []

        for pa in range(sh_a.nprim):
            for pb in range(sh_b.nprim):
                g = float(sh_a.exp[pa]) + float(sh_b.exp[pb])
                # Build product primitives in the same packed convention as
                # BasisCartSoA: coefficients multiply **unnormalized**
                # primitives exp(-g r^2), and primitive normalization is folded
                # into the coefficient. Since the AO shells are already packed
                # in this convention, the primitive product coefficient is just
                # the product of the AO primitive coefficients.
                c = float(sh_a.coef[pa]) * float(sh_b.coef[pb])
                gamma_list.append(g)
                coeff_list.append(c)

        exp_u, coef_u = _merge_duplicate_exponents(gamma_list, coeff_list)

        # Skip shells where all coefficients are negligible
        if exp_u.size == 0 or np.max(np.abs(coef_u)) < 1.0e-30:
            continue

        shells.append(RICDShell(
            atom_type_key=atom_type.key,
            l=L,
            prim_exp=exp_u,
            prim_coef=coef_u,
        ))

    return _deduplicate_shells(shells)


def make_primitive_candidate_shells(
    atom_type: AtomicBasisType,
    ao_basis: "BasisCartSoA",  # noqa: F821
    *,
    skip_high_ac: bool = False,
) -> list[RICDShell]:
    """Build the primitive (uncontracted) candidate shell pool for acCD.

    Implements design doc §7.5: every distinct exponent sum becomes its own
    single-primitive shell.

    **Important scaling detail**
    ----------------------------
    Molcas constructs the SLIM primitive-product metric from *primitive*
    (uncontracted) AO functions. To match its behavior, each primitive-product
    basis function should carry the product of the *original* primitive
    normalization factors, not the normalization of the resulting exponent-sum
    primitive. This scaling strongly affects which primitive sums survive the
    CD ("SLIM") selection.
    """
    pairs = iter_retained_shell_pairs(atom_type, skip_high_ac=skip_high_ac)
    shells: list[RICDShell] = []
    seen: set[tuple[int, int]] = set()  # (L, rounded_exp_key)

    for ia, ib in pairs:
        sh_a = _shell_view_from_meta(atom_type, ao_basis, ia)
        sh_b = _shell_view_from_meta(atom_type, ao_basis, ib)
        L = sh_a.l + sh_b.l

        # Primitive normalization factors for the *original* primitives.
        # These are used to scale primitive-product shells (Molcas-like).
        norm_a = np.asarray(
            primitive_norm_cart_like_pyscf(int(sh_a.l), np.asarray(sh_a.exp, dtype=np.float64)),
            dtype=np.float64,
        ).ravel()
        norm_b = np.asarray(
            primitive_norm_cart_like_pyscf(int(sh_b.l), np.asarray(sh_b.exp, dtype=np.float64)),
            dtype=np.float64,
        ).ravel()

        for pa in range(sh_a.nprim):
            for pb in range(sh_b.nprim):
                g = float(sh_a.exp[pa]) + float(sh_b.exp[pb])
                key = (L, _rounded_exp_key(g))
                if key in seen:
                    continue
                seen.add(key)
                # Scale by product of original primitive norms (see docstring).
                coef = float(norm_a[int(pa)] * norm_b[int(pb)])
                shells.append(RICDShell(
                    atom_type_key=atom_type.key,
                    l=L,
                    prim_exp=np.array([g], dtype=np.float64),
                    prim_coef=np.array([coef], dtype=np.float64),
                ))

    return shells


# ---------------------------------------------------------------------------
# Shell deduplication
# ---------------------------------------------------------------------------

def _shell_dedup_key(sh: RICDShell) -> tuple:
    """Hash key for deduplicating equivalent candidate shells."""
    exp_key = tuple(round(float(e) * 1e12) for e in sh.prim_exp)
    coef_key = tuple(round(float(c) * 1e12) for c in sh.prim_coef)
    return (sh.l, exp_key, coef_key)


def _deduplicate_shells(shells: list[RICDShell]) -> list[RICDShell]:
    """Remove duplicate shells (same l, exponents, and coefficients)."""
    seen: set[tuple] = set()
    out: list[RICDShell] = []
    for sh in shells:
        k = _shell_dedup_key(sh)
        if k not in seen:
            seen.add(k)
            out.append(sh)
    return out


__all__ = [
    "extract_atomic_basis_types",
    "iter_retained_shell_pairs",
    "make_contracted_candidate_shells",
    "make_primitive_candidate_shells",
    "molcas_keep_shell_from_atomic_number",
]
