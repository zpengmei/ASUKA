from __future__ import annotations

"""True Cholesky renormalization for the RICD generator.

Implements design doc §7.8: per-sector eigenvalue-based orthogonalization
of the selected auxiliary shells.  After renormalization, the sector
metric is close to the identity and the generated basis has a stable
"true Cholesky" character.

All operations are pure NumPy on small atomic-scale matrices.
"""

from collections import defaultdict

import numpy as np

from asuka.cueri.basis_cart import BasisCartSoA
from asuka.cueri.cart import ncart
from asuka.integrals.ricd_types import RICDOptions, RICDShell


# ---------------------------------------------------------------------------
# Sector grouping
# ---------------------------------------------------------------------------

def _group_shells_by_l(shells: list[RICDShell]) -> dict[int, list[int]]:
    """Group shell indices by angular momentum."""
    by_l: dict[int, list[int]] = defaultdict(list)
    for idx, sh in enumerate(shells):
        by_l[sh.l].append(idx)
    return dict(by_l)


# ---------------------------------------------------------------------------
# Common primitive set merging
# ---------------------------------------------------------------------------

def _merge_to_common_primitives(
    shells: list[RICDShell],
    *,
    rtol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge shells in a sector to a common set of primitive exponents.

    Parameters
    ----------
    shells : list[RICDShell]
        Shells that share the same angular momentum.
    rtol : float
        Relative tolerance for considering two exponents identical.

    Returns
    -------
    common_exp : np.ndarray, shape (n_common,)
        Sorted (descending) unique exponent set.
    coef_matrix : np.ndarray, shape (n_common, n_shells)
        Coefficient matrix: column *j* gives shell *j*'s coefficients
        on the common exponent set.
    """
    if not shells:
        return np.empty(0, dtype=np.float64), np.empty((0, 0), dtype=np.float64)

    # Collect all unique exponents across shells
    all_exps: list[float] = []
    for sh in shells:
        all_exps.extend(float(e) for e in sh.prim_exp)

    # Sort descending and merge duplicates
    all_exps_sorted = sorted(set(all_exps), reverse=True)
    merged_exps: list[float] = []
    for e in all_exps_sorted:
        if merged_exps and abs(e - merged_exps[-1]) <= rtol * max(abs(e), abs(merged_exps[-1]), 1.0e-30):
            continue
        merged_exps.append(e)

    common_exp = np.asarray(merged_exps, dtype=np.float64)
    n_common = len(common_exp)
    n_shells = len(shells)
    coef_matrix = np.zeros((n_common, n_shells), dtype=np.float64)

    for j, sh in enumerate(shells):
        for k in range(int(sh.prim_exp.size)):
            e_k = float(sh.prim_exp[k])
            c_k = float(sh.prim_coef[k])
            # Find matching common exponent
            for i in range(n_common):
                if abs(e_k - common_exp[i]) <= rtol * max(abs(e_k), abs(common_exp[i]), 1.0e-30):
                    coef_matrix[i, j] += c_k
                    break

    return common_exp, coef_matrix


# ---------------------------------------------------------------------------
# Coulomb metric for the renormalization transform (Molcas ReNorm2 analogue)
# ---------------------------------------------------------------------------

def _pack_one_center_shells(
    shells: list[RICDShell],
    *,
    center: np.ndarray | None = None,
) -> BasisCartSoA:
    """Pack shells at a single center into a ``BasisCartSoA``.

    Note: coefficients are used as-is (they are already in BasisCartSoA packed
    convention).
    """
    if center is None:
        center = np.zeros(3, dtype=np.float64)
    else:
        center = np.asarray(center, dtype=np.float64).reshape(3)

    shell_cxyz_list: list[np.ndarray] = []
    shell_prim_start_list: list[int] = []
    shell_nprim_list: list[int] = []
    shell_l_list: list[int] = []
    shell_ao_start_list: list[int] = []
    prim_exp_list: list[float] = []
    prim_coef_list: list[float] = []

    ao_cursor = 0
    prim_cursor = 0

    for sh in shells:
        nprim = int(sh.prim_exp.size)
        shell_cxyz_list.append(center)
        shell_prim_start_list.append(prim_cursor)
        shell_nprim_list.append(nprim)
        shell_l_list.append(int(sh.l))
        shell_ao_start_list.append(ao_cursor)

        prim_exp_list.extend(float(e) for e in sh.prim_exp)
        prim_coef_list.extend(float(c) for c in sh.prim_coef)

        ao_cursor += ncart(int(sh.l))
        prim_cursor += nprim

    if not shells:
        return BasisCartSoA(
            shell_cxyz=np.empty((0, 3), dtype=np.float64),
            shell_prim_start=np.empty(0, dtype=np.int32),
            shell_nprim=np.empty(0, dtype=np.int32),
            shell_l=np.empty(0, dtype=np.int32),
            shell_ao_start=np.empty(0, dtype=np.int32),
            prim_exp=np.empty(0, dtype=np.float64),
            prim_coef=np.empty(0, dtype=np.float64),
        )

    return BasisCartSoA(
        shell_cxyz=np.array(shell_cxyz_list, dtype=np.float64).reshape(-1, 3),
        shell_prim_start=np.array(shell_prim_start_list, dtype=np.int32),
        shell_nprim=np.array(shell_nprim_list, dtype=np.int32),
        shell_l=np.array(shell_l_list, dtype=np.int32),
        shell_ao_start=np.array(shell_ao_start_list, dtype=np.int32),
        prim_exp=np.array(prim_exp_list, dtype=np.float64),
        prim_coef=np.array(prim_coef_list, dtype=np.float64),
    )


def _coulomb_metric_shell_representatives(
    shells: list[RICDShell],
    *,
    threads: int = 0,
) -> np.ndarray:
    """Coulomb metric between shells using one representative Cartesian component per shell.

    This mirrors the Molcas ReNorm2 intent: build a small shell-level metric that
    mixes *radial contractions* (not arbitrary Cartesian-component mixtures).
    """
    from asuka.integrals.cueri_df_cpu import metric_2c2e_basis_cpu  # noqa: PLC0415

    basis = _pack_one_center_shells(shells)
    V = metric_2c2e_basis_cpu(basis, threads=int(threads))

    # Representative AO index per shell: first Cartesian component of each shell.
    rep = np.asarray(basis.shell_ao_start, dtype=np.int64).ravel()
    if rep.size == 0:
        return np.empty((0, 0), dtype=np.float64)

    M = V[np.ix_(rep, rep)]
    return 0.5 * (M + M.T)


# ---------------------------------------------------------------------------
# Core renormalization
# ---------------------------------------------------------------------------

def renorm_sector(
    shells: list[RICDShell],
    *,
    threshold_cb: float,
    threads: int = 0,
) -> list[RICDShell]:
    """Renormalize a single angular-momentum sector into a true Cholesky basis.

    Parameters
    ----------
    shells : list[RICDShell]
        Selected shells for this sector (all with the same ``l``).
    threshold_cb : float
        Eigenvalue cutoff: ``max(renorm_abs_floor, renorm_rel_factor * threshold)``.

    Returns
    -------
    list[RICDShell]
        Renormalized shells.  The number may be less than *len(shells)*
        if some eigenvalues are discarded.
    """
    if not shells:
        return []

    L = shells[0].l
    atom_key = shells[0].atom_type_key
    n_shells = len(shells)

    # Coulomb metric at the shell level (Molcas ReNorm2 analogue)
    M = _coulomb_metric_shell_representatives(shells, threads=int(threads))
    if M.size == 0:
        return []

    # Eigendecompose: M = U Λ U^T
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # Discard small eigenvalues
    keep_mask = eigenvalues > threshold_cb
    n_keep = int(np.sum(keep_mask))
    if n_keep == 0:
        return []

    lam_kept = eigenvalues[keep_mask]
    U_kept = eigenvectors[:, keep_mask]  # (n_shells, n_keep)

    # Transform: T = U Λ^{-1/2}
    T = U_kept * (1.0 / np.sqrt(lam_kept))[np.newaxis, :]  # (n_shells, n_keep)

    # Merge to common primitives and apply the shell-mixing transform.
    common_exp, coef_matrix = _merge_to_common_primitives(shells)
    if common_exp.size == 0 or coef_matrix.shape[1] == 0:
        return []

    # New coefficient matrix: C_new = C_old @ T → (n_common, n_keep)
    C_new = coef_matrix @ T

    # Build new shells
    result: list[RICDShell] = []
    for j in range(n_keep):
        col = C_new[:, j]
        # Drop negligible primitives
        mask = np.abs(col) > 1.0e-30
        if not np.any(mask):
            continue
        result.append(RICDShell(
            atom_type_key=atom_key,
            l=L,
            prim_exp=np.asarray(common_exp[mask], dtype=np.float64).copy(),
            prim_coef=np.asarray(col[mask], dtype=np.float64).copy(),
        ))

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def renorm_true_cholesky(
    shells: list[RICDShell],
    *,
    threshold_cd: float,
    options: RICDOptions,
    threads: int = 0,
) -> list[RICDShell]:
    """Renormalize selected RICD shells into a true Cholesky basis.

    Applies per-sector eigendecomposition and coefficient transform
    as described in design doc §7.8.

    Parameters
    ----------
    shells : list[RICDShell]
        Selected shells (may have mixed angular momenta).
    threshold_cd : float
        The per-type CD threshold ``τ_t``.
    options : RICDOptions
        Generator options (for ``renorm_rel_factor`` and ``renorm_abs_floor``).

    Returns
    -------
    list[RICDShell]
        Final renormalized shells.
    """
    threshold_cb = max(
        float(options.renorm_abs_floor),
        float(options.renorm_rel_factor) * float(threshold_cd),
    )

    by_l = _group_shells_by_l(shells)

    result: list[RICDShell] = []
    for L in sorted(by_l.keys()):
        sector_indices = by_l[L]
        sector_shells = [shells[i] for i in sector_indices]
        result.extend(renorm_sector(sector_shells, threshold_cb=threshold_cb, threads=int(threads)))

    return result


__all__ = ["renorm_true_cholesky"]
