from __future__ import annotations

"""Public builder for RICD (Resolution of Identity Cholesky Decomposition) auxiliary bases.

This module provides ``build_ricd_aux_basis()``, which generates a Cartesian
auxiliary basis from the orbital basis itself using the aCD or acCD procedure.
The result is a standard ``BasisCartSoA`` that plugs into ASUKA's existing
molecular DF pipeline.

Algorithm overview (design doc §12.1):

1. Extract unique atomic basis types from the molecular AO basis.
2. Per atom type:
   a. Build contracted candidate shells (product shell pool).
   b. Compute the one-center Coulomb metric via the cuERI CPU engine.
   c. Run shell-block pivoted Cholesky to select aCD shells.
   d. (acCD only) Build primitive pool, run Cholesky, project aCD → slim.
   e. Renormalize into a true Cholesky basis.
3. Replicate per-atom-type shells for every atom in the molecule.
4. Pack into a molecular ``BasisCartSoA``.
"""

import time
from collections import defaultdict
import os
from typing import Any

import numpy as np

from asuka.cueri.basis_cart import BasisCartSoA
from asuka.cueri.cart import ncart
from asuka.integrals.int1e_cart import nao_cart_from_basis
from asuka.integrals.ricd_atomic import (
    extract_atomic_basis_types,
    make_contracted_candidate_shells,
    make_primitive_candidate_shells,
)
from asuka.integrals.ricd_cd import block_pivoted_cholesky, build_shell_blocks
from asuka.integrals.ricd_renorm import renorm_true_cholesky
from asuka.integrals.ricd_types import (
    AtomicBasisType,
    RICDGeneratedBasis,
    RICDOptions,
    RICDShell,
)


# ---------------------------------------------------------------------------
# Per-type threshold scaling
# ---------------------------------------------------------------------------

def _type_scale(opts: RICDOptions, type_key: str) -> float:
    """Return the per-atom-type threshold multiplier."""
    if opts.type_threshold_scale is not None and type_key in opts.type_threshold_scale:
        return float(opts.type_threshold_scale[type_key])
    return 1.0


# ---------------------------------------------------------------------------
# Packing helpers: RICD shells → BasisCartSoA
# ---------------------------------------------------------------------------

def _pack_one_center_shells(
    shells: list[RICDShell],
    center: np.ndarray | None = None,
) -> BasisCartSoA:
    """Pack a list of RICD shells located at one center into a ``BasisCartSoA``.

    The shells' ``prim_coef`` values are used *as-is* — they are already in
    BasisCartSoA packed convention (include primitive normalization).
    No additional normalization is applied.

    Parameters
    ----------
    shells : list[RICDShell]
        Shells to pack (all placed at the same center).
    center : np.ndarray or None
        (3,) coordinate in Bohr. Defaults to origin.
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


def _concat_packed_bases(a: BasisCartSoA, b: BasisCartSoA) -> BasisCartSoA:
    """Concatenate two packed bases at the same center into one.

    Shell indices and AO offsets from *b* are shifted to follow *a*.
    """
    nao_a = nao_cart_from_basis(a)
    nprim_a = int(np.asarray(a.prim_exp, dtype=np.float64).shape[0])

    return BasisCartSoA(
        shell_cxyz=np.concatenate([
            np.asarray(a.shell_cxyz, dtype=np.float64),
            np.asarray(b.shell_cxyz, dtype=np.float64),
        ], axis=0),
        shell_prim_start=np.concatenate([
            np.asarray(a.shell_prim_start, dtype=np.int32),
            np.asarray(b.shell_prim_start, dtype=np.int32) + np.int32(nprim_a),
        ], axis=0),
        shell_nprim=np.concatenate([
            np.asarray(a.shell_nprim, dtype=np.int32),
            np.asarray(b.shell_nprim, dtype=np.int32),
        ], axis=0),
        shell_l=np.concatenate([
            np.asarray(a.shell_l, dtype=np.int32),
            np.asarray(b.shell_l, dtype=np.int32),
        ], axis=0),
        shell_ao_start=np.concatenate([
            np.asarray(a.shell_ao_start, dtype=np.int32),
            np.asarray(b.shell_ao_start, dtype=np.int32) + np.int32(nao_a),
        ], axis=0),
        prim_exp=np.concatenate([
            np.asarray(a.prim_exp, dtype=np.float64),
            np.asarray(b.prim_exp, dtype=np.float64),
        ], axis=0),
        prim_coef=np.concatenate([
            np.asarray(a.prim_coef, dtype=np.float64),
            np.asarray(b.prim_coef, dtype=np.float64),
        ], axis=0),
    )


# ---------------------------------------------------------------------------
# acCD projection (design doc §12.4)
# ---------------------------------------------------------------------------

def _common_angular_sectors(
    shells_a: list[RICDShell],
    shells_b: list[RICDShell],
) -> set[int]:
    """Angular momentum values present in both shell lists."""
    ls_a = {sh.l for sh in shells_a}
    ls_b = {sh.l for sh in shells_b}
    return ls_a & ls_b


def _recombine_slim_shells_with_coeff_matrix(
    slim_shells: list[RICDShell],
    C: np.ndarray,
) -> list[RICDShell]:
    """Build contracted shells from slim primitives and coefficient matrix.

    Parameters
    ----------
    slim_shells : list[RICDShell]
        The selected slim (one-primitive) shells for this sector.
    C : np.ndarray, shape (n_slim_shell, n_acd_shell)
        Coulomb-metric projection coefficient matrix at the **shell** level,
        where columns represent the projected acCD shells in the slim primitive
        span.

    Returns
    -------
    list[RICDShell]
        One RICDShell per column of *C*, with primitives from the slim set.
    """
    if not slim_shells or C.size == 0:
        return []

    # All slim shells share the same L and atom_type_key.
    L = slim_shells[0].l
    atom_key = slim_shells[0].atom_type_key

    slim_exp = np.asarray([float(sh.prim_exp[0]) for sh in slim_shells], dtype=np.float64)
    slim_coef = np.asarray([float(sh.prim_coef[0]) for sh in slim_shells], dtype=np.float64)
    if int(C.shape[0]) != int(slim_exp.size):
        raise ValueError("C first dimension must match number of slim shells")

    result: list[RICDShell] = []
    n_cols = int(C.shape[1])
    for col_j in range(n_cols):
        shell_coefs = np.asarray(C[:, col_j], dtype=np.float64).ravel() * slim_coef
        mask = np.abs(shell_coefs) > 1.0e-30
        if not np.any(mask):
            continue

        exp_j = slim_exp[mask]
        coef_j = shell_coefs[mask]
        order = np.argsort(exp_j)[::-1]
        exp_j = exp_j[order]
        coef_j = coef_j[order]

        result.append(RICDShell(
            atom_type_key=atom_key,
            l=L,
            prim_exp=exp_j.copy(),
            prim_coef=coef_j.copy(),
        ))

    return result


def _project_contracted_into_slim(
    shells_acd: list[RICDShell],
    shells_slim: list[RICDShell],
    *,
    threads: int = 0,
) -> list[RICDShell]:
    """Project contracted aCD shells into the slim primitive span (design doc §12.4).

    Works sector-by-sector (per angular momentum L).  For each sector:

    1. Pack slim and aCD shells into one-center packed bases.
    2. Concatenate: ``basis_both = [slim | acd]``.
    3. Compute the Coulomb metric ``V = (both|both)`` via cuERI CPU.
    4. Extract shell-level matrices using one representative Cartesian component
       per shell: ``G = (slim|slim)``, ``K = (slim|acd)``.
    5. Solve ``C = G^{-1} K`` (Coulomb-metric least-squares).
    6. Recombine slim shells with coefficient matrix C to produce one projected
       shell per aCD shell.
    """
    from asuka.integrals.cueri_df_cpu import metric_2c2e_basis_cpu  # noqa: PLC0415

    sectors = _common_angular_sectors(shells_acd, shells_slim)
    # Also include sectors in acd that have no slim coverage (these pass through
    # as-is, though renorm will handle them downstream).
    acd_only_sectors = {sh.l for sh in shells_acd} - sectors

    result: list[RICDShell] = []

    for L in sorted(sectors):
        slim_L = [sh for sh in shells_slim if sh.l == L]
        acd_L = [sh for sh in shells_acd if sh.l == L]

        if not slim_L or not acd_L:
            result.extend(acd_L)
            continue

        basis_slim = _pack_one_center_shells(slim_L)
        basis_acd = _pack_one_center_shells(acd_L)
        basis_both = _concat_packed_bases(basis_slim, basis_acd)

        V = metric_2c2e_basis_cpu(basis_both, threads=threads)
        n_slim_shell = int(np.asarray(basis_slim.shell_l, dtype=np.int32).shape[0])
        n_acd_shell = int(np.asarray(basis_acd.shell_l, dtype=np.int32).shape[0])

        # Representative AO index per shell: first Cartesian component.
        sh_ao_start = np.asarray(basis_both.shell_ao_start, dtype=np.int64).ravel()
        rep_slim = sh_ao_start[:n_slim_shell]
        rep_acd = sh_ao_start[n_slim_shell : n_slim_shell + n_acd_shell]

        G = V[np.ix_(rep_slim, rep_slim)]
        K = V[np.ix_(rep_slim, rep_acd)]

        # Solve G C = K → C = G^{-1} K
        # G is symmetric positive definite; use Cholesky for stability.
        try:
            L_chol = np.linalg.cholesky(G)
            # C = inv(G) K = inv(L^T) inv(L) K
            Y = np.linalg.solve(L_chol, K)
            C = np.linalg.solve(L_chol.T, Y)
        except np.linalg.LinAlgError:
            # Fallback: use pseudoinverse if G is singular.
            C = np.linalg.lstsq(G, K, rcond=None)[0]

        result.extend(_recombine_slim_shells_with_coeff_matrix(slim_L, C))

    # Pass through aCD shells in sectors with no slim coverage.
    for L in sorted(acd_only_sectors):
        result.extend(sh for sh in shells_acd if sh.l == L)

    return result


# ---------------------------------------------------------------------------
# Slim primitive selection with rank retry (design doc §7.6)
# ---------------------------------------------------------------------------

def _count_shells_by_l(shells: list[RICDShell]) -> dict[int, int]:
    """Count shells per angular momentum."""
    counts: dict[int, int] = defaultdict(int)
    for sh in shells:
        counts[sh.l] += 1
    return dict(counts)


def _select_slim_primitive_shells_with_rank_retry(
    primitive_shells: list[RICDShell],
    shells_acd: list[RICDShell],
    tau_p: float,
    opts: RICDOptions,
    *,
    threads: int = 0,
) -> list[int]:
    """Run shell-block CD on the primitive pool with rank safeguard retries.

    Per §7.6: if in any angular-momentum sector the number of selected
    primitive shells is less than the number of contracted aCD shells in
    that sector, halve ``tau_p`` and retry, up to ``primitive_retry_halves``
    times.

    Returns indices into *primitive_shells* of the selected primitives.
    """
    from asuka.integrals.cueri_df_cpu import metric_2c2e_basis_cpu  # noqa: PLC0415

    acd_counts = _count_shells_by_l(shells_acd)

    # IMPORTANT: select slim primitives **per angular-momentum sector**.
    #
    # A global CD across mixed-L primitives can (and does) eliminate whole
    # sectors via cross-L coupling in the Cartesian representation, causing
    # the subsequent per-L reprojection to lose essential tight primitives
    # (notably core-valence and core-core radial content). Molcas performs
    # primitive slimming in a more localized way; the per-L selection is the
    # robust ASUKA analogue.
    keep_all: list[int] = []

    for L in sorted(acd_counts.keys()):
        need = int(acd_counts.get(int(L), 0))
        if need <= 0:
            continue

        idxs = [i for i, sh in enumerate(primitive_shells) if int(sh.l) == int(L)]
        if not idxs:
            continue

        prim_L = [primitive_shells[i] for i in idxs]
        basis_L = _pack_one_center_shells(prim_L)
        Mp_L = metric_2c2e_basis_cpu(basis_L, threads=int(threads))
        blocks_L = build_shell_blocks(prim_L)

        tau_cur = float(tau_p)
        keep_L: list[int] = []
        for _retry in range(int(opts.primitive_retry_halves) + 1):
            keep_L = block_pivoted_cholesky(Mp_L, blocks_L, tau_cur)
            if int(len(keep_L)) >= int(need):
                break
            tau_cur *= 0.5

        # If we still failed to reach the required rank, fall back to selecting
        # the leading `need` primitives by score order (deterministic) to keep
        # the subsequent projection well-posed.
        if int(len(keep_L)) < int(need):
            keep_L = list(range(min(int(need), int(len(prim_L)))))

        keep_all.extend(int(idxs[i]) for i in keep_L)

    # De-duplicate while preserving order.
    seen: set[int] = set()
    keep_unique: list[int] = []
    for i in keep_all:
        if int(i) in seen:
            continue
        seen.add(int(i))
        keep_unique.append(int(i))

    return keep_unique


# ---------------------------------------------------------------------------
# Molecular basis packing: replicate per atom type
# ---------------------------------------------------------------------------

def _replicate_and_pack_generated_shells(
    generated_by_type: dict[str, list[RICDShell]],
    atom_types: list[AtomicBasisType],
    atoms_bohr: list[tuple[str, np.ndarray]],
) -> BasisCartSoA:
    """Replicate generated shells for each atom and pack into molecular ``BasisCartSoA``.

    For each atom in the molecule, look up its type key, copy the generated
    auxiliary shells, and place them at that atom's center.  The result is a
    molecular auxiliary basis in standard ``BasisCartSoA`` format.
    """
    # Build type_key → generated shells lookup.
    type_key_for_atom: dict[int, str] = {}
    for at in atom_types:
        for iatom in at.atom_indices:
            type_key_for_atom[iatom] = at.key

    n_atoms = len(atoms_bohr)

    shell_cxyz_list: list[np.ndarray] = []
    shell_prim_start_list: list[int] = []
    shell_nprim_list: list[int] = []
    shell_l_list: list[int] = []
    shell_ao_start_list: list[int] = []
    prim_exp_list: list[float] = []
    prim_coef_list: list[float] = []

    ao_cursor = 0
    prim_cursor = 0

    for iatom in range(n_atoms):
        sym, xyz = atoms_bohr[iatom]
        center = np.asarray(xyz, dtype=np.float64).reshape(3)
        type_key = type_key_for_atom.get(iatom)
        if type_key is None:
            continue
        shells = generated_by_type.get(type_key, [])

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

    if not shell_cxyz_list:
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


# ---------------------------------------------------------------------------
# Name summary
# ---------------------------------------------------------------------------

def _summarize_ricd_name(opts: RICDOptions, stats: dict[str, Any]) -> str:
    """Build a human-readable summary string for the generated basis."""
    mode = str(opts.mode)
    tau = float(opts.threshold)
    n_types = len(stats.get("atom_types", {}))
    total_final = sum(
        int(v.get("n_final", 0))
        for v in stats.get("atom_types", {}).values()
    )
    return f"{mode}/tau={tau:.1e}/{n_types}types/{total_final}shells"


# ---------------------------------------------------------------------------
# Public builder API (design doc §12.1)
# ---------------------------------------------------------------------------

def build_ricd_aux_basis(
    ao_basis: BasisCartSoA,
    atoms_bohr: list[tuple[str, np.ndarray]],
    *,
    options: RICDOptions | None = None,
    threads: int = 0,
    profile: dict | None = None,
) -> RICDGeneratedBasis:
    """Generate a RICD auxiliary basis from the orbital basis.

    Parameters
    ----------
    ao_basis : BasisCartSoA
        Molecular AO basis (expanded, Cartesian).
    atoms_bohr : list of (symbol, xyz)
        Atoms with coordinates in Bohr.
    options : RICDOptions or None
        RICD generator options.  ``None`` uses defaults (acCD, τ=1e-4).
    threads : int
        Number of CPU threads for metric evaluation (0 = auto).
    profile : dict or None
        If not None, timing and statistics are stored here.

    Returns
    -------
    RICDGeneratedBasis
        The generated auxiliary basis (packed as ``BasisCartSoA``) and metadata.
    """
    from asuka.integrals.cueri_df_cpu import metric_2c2e_basis_cpu  # noqa: PLC0415

    opts = RICDOptions() if options is None else options

    t_total_start = time.perf_counter()

    # Step 0: identify unique atomic AO basis types.
    atom_types = extract_atomic_basis_types(ao_basis, atoms_bohr)

    generated_by_type: dict[str, list[RICDShell]] = {}
    stats: dict[str, Any] = {
        "atom_types": {},
        "mode": str(opts.mode),
        "threshold": float(opts.threshold),
    }

    for at in atom_types:
        t_at_start = time.perf_counter()
        tau_t = float(opts.threshold) * _type_scale(opts, at.key)

        # Optional per-type on-disk cache (Molcas RICDLib-like)
        cache_hit = False
        cache_meta: dict[str, Any] | None = None
        shells_final: list[RICDShell] | None = None
        cache_path = None
        if bool(opts.cache):
            try:  # best-effort: never fail due to cache IO
                from asuka.integrals.ricd_cache import (  # noqa: PLC0415
                    load_ricd_type_cache,
                    ricd_type_cache_path,
                )

                cache_path = ricd_type_cache_path(at.key, tau_t=float(tau_t), opts=opts)
                hit = load_ricd_type_cache(at.key, tau_t=float(tau_t), opts=opts)
                if hit is not None:
                    shells_final, cache_meta = hit
                    cache_hit = True
            except Exception:
                cache_hit = False
                cache_meta = None
                shells_final = None

        if cache_hit and shells_final is not None:
            generated_by_type[at.key] = shells_final
            cached_stats = {}
            try:
                if isinstance(cache_meta, dict):
                    cached_stats_raw = cache_meta.get("stats", None)
                    if isinstance(cached_stats_raw, dict):
                        cached_stats = dict(cached_stats_raw)
            except Exception:
                cached_stats = {}

            stats["atom_types"][at.key] = {
                "cache_hit": True,
                "cache_path": None if cache_path is None else os.fspath(cache_path),
                "n_atoms": len(at.atom_indices),
                "n_contracted_candidates": int(cached_stats.get("n_contracted_candidates", 0)),
                "n_selected_acd": int(cached_stats.get("n_selected_acd", 0)),
                "n_final": len(shells_final),
                "time_s": float(time.perf_counter() - t_at_start),
            }
            continue

        # Step 1: contracted candidate shell pool (aCD pool).
        contracted_shells = make_contracted_candidate_shells(
            at, ao_basis, skip_high_ac=opts.skip_high_ac,
        )

        if not contracted_shells:
            generated_by_type[at.key] = []
            stats["atom_types"][at.key] = {
                "cache_hit": False,
                "cache_path": None if cache_path is None else os.fspath(cache_path),
                "n_atoms": len(at.atom_indices),
                "n_contracted_candidates": 0,
                "n_selected_acd": 0,
                "n_final": 0,
                "time_s": float(time.perf_counter() - t_at_start),
            }
            continue

        # Step 2: one-center Coulomb metric of the contracted pool.
        basis_contracted = _pack_one_center_shells(contracted_shells, center=at.rep_center)
        Mc = metric_2c2e_basis_cpu(basis_contracted, threads=threads)

        # Step 3: shell-block pivoted Cholesky (aCD selection).
        c_blocks = build_shell_blocks(contracted_shells)
        keep_c = block_pivoted_cholesky(Mc, c_blocks, threshold=tau_t)
        shells_acd = [contracted_shells[i] for i in keep_c]

        if opts.mode == "acd":
            # aCD mode: renormalize directly.
            shells_final = renorm_true_cholesky(
                shells_acd, threshold_cd=tau_t, options=opts, threads=int(threads),
            )
        else:
            # acCD mode: primitive pool → slim selection → projection → renorm.

            # Step 4: primitive candidate shell pool.
            primitive_shells = make_primitive_candidate_shells(
                at, ao_basis, skip_high_ac=opts.skip_high_ac,
            )

            if not primitive_shells:
                # Fallback: renormalize aCD shells directly.
                shells_final = renorm_true_cholesky(
                    shells_acd, threshold_cd=tau_t, options=opts, threads=int(threads),
                )
            else:
                # Step 5: primitive metric + slim selection with rank retry.
                tau_p = float(opts.primitive_threshold_ratio) * tau_t
                keep_p = _select_slim_primitive_shells_with_rank_retry(
                    primitive_shells, shells_acd, tau_p, opts, threads=int(threads),
                )
                shells_slim = [primitive_shells[i] for i in keep_p]

                if not shells_slim:
                    shells_final = renorm_true_cholesky(
                        shells_acd, threshold_cd=tau_t, options=opts, threads=int(threads),
                    )
                else:
                    # Step 6: project aCD into slim primitive span.
                    shells_proj = _project_contracted_into_slim(
                        shells_acd, shells_slim, threads=threads,
                    )

                    # Step 7: renormalize.
                    shells_final = renorm_true_cholesky(
                        shells_proj, threshold_cd=tau_t, options=opts, threads=int(threads),
                    )

        generated_by_type[at.key] = shells_final
        stats["atom_types"][at.key] = {
            "cache_hit": False,
            "cache_path": None if cache_path is None else os.fspath(cache_path),
            "n_atoms": len(at.atom_indices),
            "n_contracted_candidates": len(contracted_shells),
            "n_selected_acd": len(shells_acd),
            "n_final": len(shells_final),
            "time_s": float(time.perf_counter() - t_at_start),
        }

        if bool(opts.cache):
            try:  # best-effort: never fail due to cache IO
                from asuka.integrals.ricd_cache import save_ricd_type_cache  # noqa: PLC0415

                save_ricd_type_cache(
                    at.key,
                    shells_final,
                    tau_t=float(tau_t),
                    opts=opts,
                    stats={
                        "n_contracted_candidates": int(len(contracted_shells)),
                        "n_selected_acd": int(len(shells_acd)),
                        "n_final": int(len(shells_final)),
                    },
                )
            except Exception:
                pass

    # Step 8: replicate per atom type and pack molecular auxiliary basis.
    packed_aux = _replicate_and_pack_generated_shells(
        generated_by_type, atom_types, atoms_bohr,
    )

    basis_name = _summarize_ricd_name(opts, stats)

    if profile is not None:
        profile["ricd_stats"] = stats
        profile["ricd_total_time_s"] = float(time.perf_counter() - t_total_start)

    return RICDGeneratedBasis(
        packed_basis=packed_aux,
        basis_name=basis_name,
        options=opts,
        stats=stats,
        atom_type_keys=tuple(generated_by_type.keys()),
    )


__all__ = ["build_ricd_aux_basis"]
