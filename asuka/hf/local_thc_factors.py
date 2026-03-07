from __future__ import annotations

"""Local-THC factor construction (GPU-first).

This module implements a pragmatic local-THC construction inspired by:
  Song & Martinez, J. Chem. Phys. 146, 034104 (2017).

Key idea
--------
Partition atoms into blocks. For each block, build THC factors (X,Z) on a
local fitting region (subset of atoms), and only for a local AO set consisting
of:
- primary AOs on block atoms
- secondary AOs on nearby atoms in other blocks (overlap-filtered)

The THC central metric Z can be built either by the auxiliary-metric fit
(default; consistent with recent THC-SCF constructions) or by a stable
aux-metric inverse projection (fallback).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.cueri.basis_subset import subset_cart_basis_by_shells
from asuka.hf.local_thc_config import LocalTHCConfig
from asuka.hf.local_thc_partition import (
    AtomBlocks,
    build_atom_blocks,
    build_atom_neighbors_by_schwarz,
    map_shells_to_atoms,
    union_neighbors,
)


@dataclass(frozen=True)
class LocalTHCBlock:
    block_id: int
    # Local AO ordering is: [early secondary][primary][late secondary]
    ao_idx_global: np.ndarray  # int32, shape (nlocal_ao,)
    n_early: int
    n_primary: int
    atoms_primary: tuple[int, ...]
    atoms_secondary_early: tuple[int, ...]
    atoms_secondary_late: tuple[int, ...]
    atoms_aux: tuple[int, ...]
    X: Any  # cupy.ndarray, (npt, nlocal_ao)
    Y: Any  # cupy.ndarray, (npt, naux) such that Z = Y @ Y.T for this block
    Z: Any | None  # optional: cupy.ndarray, (npt, npt)
    points: Any  # cupy.ndarray, (npt, 3)
    weights: Any  # cupy.ndarray, (npt,)
    L_metric: Any  # cupy.ndarray, (naux, naux) (Cholesky of aux metric for this block)
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class LocalTHCFactors:
    blocks: tuple[LocalTHCBlock, ...]
    nao: int
    ao_rep: str  # 'cart' or 'sph'
    L_metric_full: Any | None = None  # optional: full aux metric Cholesky (for DF warmup)
    meta: dict[str, Any] | None = None


def _require_cupy():
    try:
        import cupy as cp  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Local-THC requires CuPy") from e
    return cp


def _downselect_by_weight(cp, points, weights, *, npt: int, return_idx: bool = False):
    points = cp.asarray(points, dtype=cp.float64)
    weights = cp.asarray(weights, dtype=cp.float64).ravel()
    if points.ndim != 2 or int(points.shape[1]) != 3:
        raise ValueError("points must have shape (npt,3)")
    if int(weights.shape[0]) != int(points.shape[0]):
        raise ValueError("weights must have shape (npt,)")

    n_all = int(weights.shape[0])
    npt = int(npt)
    if npt <= 0:
        raise ValueError("npt must be > 0")
    if npt >= n_all:
        pts = cp.ascontiguousarray(points)
        w = cp.ascontiguousarray(weights)
        if return_idx:
            idx = cp.arange(int(n_all), dtype=cp.int64)
            return pts, w, cp.ascontiguousarray(idx)
        return pts, w

    w_all_sum = cp.sum(weights)
    idx = cp.argpartition(weights, int(n_all - npt))[int(n_all - npt) :]
    w_sel = weights[idx]
    try:
        order = cp.lexsort((idx, -w_sel))
        idx = idx[order]
    except Exception:  # pragma: no cover
        idx_h = cp.asnumpy(idx)
        w_h = cp.asnumpy(w_sel)
        order_h = np.lexsort((idx_h, -w_h))
        idx = cp.asarray(idx_h[order_h], dtype=cp.int64)

    pts = points[idx]
    w = weights[idx]
    w_sum = cp.sum(w)
    if float(w_sum) != 0.0:
        w = w * (w_all_sum / w_sum)
    pts = cp.ascontiguousarray(pts)
    w = cp.ascontiguousarray(w)
    if return_idx:
        return pts, w, cp.ascontiguousarray(idx)
    return pts, w


def _make_atom_grids_device(
    mol_or_coords: Any,
    *,
    grid_kind: str,
    dvr_basis: Any | None,
    grid_options: dict[str, Any] | None,
    grid_spec: Any,
    stream: Any,
):
    """Return per-atom device grids as lists of CuPy arrays.

    This uses the unified density-layer dispatcher so grid logic lives in one place.
    """

    cp = _require_cupy()
    grid_kind_s = str(grid_kind).strip().lower()
    grid_opts = {} if grid_options is None else dict(grid_options)

    from asuka.density import GridRequest, iter_grid  # noqa: PLC0415
    from asuka.density.grids import _coords_bohr  # noqa: PLC0415

    coords = np.asarray(_coords_bohr(mol_or_coords), dtype=np.float64).reshape((-1, 3))
    natm = int(coords.shape[0])
    pts_atoms: list[Any] = [cp.zeros((0, 3), dtype=cp.float64) for _ in range(natm)]
    w_atoms: list[Any] = [cp.zeros((0,), dtype=cp.float64) for _ in range(natm)]

    if grid_kind_s in {"fdvr", "f-dvr", "f_dvr"}:
        raise NotImplementedError("F-DVR is a global/molecular grid and is not supported for local-THC")

    if grid_kind_s in {"becke", "becke_grid", "atom", "atom-centered"}:
        kind = "becke"
        angular_prune = bool(grid_opts.get("angular_prune", False))
    elif grid_kind_s in {"rdvr", "r-dvr", "r_dvr"}:
        kind = "rdvr"
        angular_prune = bool(grid_opts.get("angular_prune", True))
        if dvr_basis is None:
            raise ValueError("grid_kind='rdvr' requires dvr_basis (use aux_basis or provide a separate DVR basis)")
    else:
        raise ValueError("grid_kind must be one of: 'becke', 'rdvr'")

    req = GridRequest(
        kind=kind,  # type: ignore[arg-type]
        backend="cuda",
        radial_n=int(getattr(grid_spec, "radial_n", 50)),
        angular_n=int(getattr(grid_spec, "angular_n", 302)),
        angular_kind=str(getattr(grid_spec, "angular_kind", "auto")),
        rmax=float(getattr(grid_spec, "rmax", 20.0)),
        becke_n=int(getattr(grid_spec, "becke_n", 3)),
        block_size=10**9,  # one batch per atom
        prune_tol=float(getattr(grid_spec, "prune_tol", 1e-16)),
        threads=int(getattr(grid_spec, "threads", 256)),
        stream=stream,
        radial_rmax=float(grid_opts.get("radial_rmax", getattr(grid_spec, "rmax", 20.0))),
        ortho_cutoff=float(grid_opts.get("ortho_cutoff", 1e-10)),
        angular_prune=bool(angular_prune),
        atom_Z=grid_opts.get("atom_Z", None),
    )

    for batch in iter_grid(mol_or_coords, request=req, dvr_basis=dvr_basis):
        ia = batch.meta.get("atom", None)
        if ia is None:
            # Best-effort fallback if meta is absent.
            pa = getattr(batch, "point_atom", None)
            if pa is None or int(getattr(pa, "size", 0)) == 0:
                continue
            try:
                ia = int(pa[0])
            except Exception:  # pragma: no cover
                ia = int(cp.asnumpy(pa[0]).item())
        ia = int(ia)
        if ia < 0 or ia >= int(natm):
            raise RuntimeError("grid batch atom index out of range")

        pts_i = cp.ascontiguousarray(cp.asarray(batch.points, dtype=cp.float64))
        w_i = cp.ascontiguousarray(cp.asarray(batch.weights, dtype=cp.float64).ravel())
        if int(pts_atoms[int(ia)].shape[0]) == 0:
            pts_atoms[int(ia)] = pts_i
            w_atoms[int(ia)] = w_i
        else:
            pts_atoms[int(ia)] = cp.ascontiguousarray(cp.concatenate([pts_atoms[int(ia)], pts_i], axis=0))
            w_atoms[int(ia)] = cp.ascontiguousarray(cp.concatenate([w_atoms[int(ia)], w_i], axis=0)).ravel()

    return pts_atoms, w_atoms


def _shells_for_atoms(atom_to_shells: list[list[int]], atoms: list[int]) -> list[int]:
    out: list[int] = []
    for ia in atoms:
        out.extend(atom_to_shells[int(ia)])
    return out


def _ao_count_per_atom(shell_l: np.ndarray, atom_to_shells: list[list[int]], *, ao_rep: str) -> np.ndarray:
    shell_l = np.asarray(shell_l, dtype=np.int32).ravel()
    natm = int(len(atom_to_shells))
    out = np.zeros((natm,), dtype=np.int64)
    ao_rep_s = str(ao_rep).strip().lower()
    if ao_rep_s not in {"cart", "sph"}:
        raise ValueError("ao_rep must be 'cart' or 'sph'")

    if ao_rep_s == "cart":
        from asuka.cueri.cart import ncart  # noqa: PLC0415

        nfn = [int(ncart(int(l))) for l in shell_l.tolist()]
    else:
        from asuka.cueri.sph import nsph  # noqa: PLC0415

        nfn = [int(nsph(int(l))) for l in shell_l.tolist()]

    for ia in range(natm):
        sids = atom_to_shells[ia]
        out[ia] = int(sum(nfn[int(s)] for s in sids))
    return out


def _shell_ao_starts_scf(ao_basis, *, ao_rep: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (shell_start, shell_nfn, nao_total) for the SCF AO representation."""

    ao_rep_s = str(ao_rep).strip().lower()
    shell_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    nshell = int(shell_l.size)
    if ao_rep_s == "cart":
        from asuka.cueri.cart import ncart  # noqa: PLC0415

        shell_start = np.asarray(ao_basis.shell_ao_start, dtype=np.int32).ravel()
        shell_nfn = np.asarray([ncart(int(l)) for l in shell_l.tolist()], dtype=np.int32)
        nao = int(0 if nshell == 0 else np.max(shell_start + shell_nfn))
        return shell_start, shell_nfn, int(nao)

    if ao_rep_s == "sph":
        from asuka.integrals.cart2sph import compute_sph_layout_from_cart_basis  # noqa: PLC0415
        from asuka.cueri.sph import nsph  # noqa: PLC0415

        shell_start_sph, nao_sph = compute_sph_layout_from_cart_basis(ao_basis)
        shell_nfn = np.asarray([nsph(int(l)) for l in shell_l.tolist()], dtype=np.int32)
        return np.asarray(shell_start_sph, dtype=np.int32).ravel(), shell_nfn, int(nao_sph)

    raise ValueError("ao_rep must be 'cart' or 'sph'")


def _atoms_to_ao_indices(
    atoms: list[int],
    *,
    atom_to_shells: list[list[int]],
    shell_start: np.ndarray,
    shell_nfn: np.ndarray,
) -> list[int]:
    out: list[int] = []
    for ia in atoms:
        for sh in atom_to_shells[int(ia)]:
            s0 = int(shell_start[int(sh)])
            n = int(shell_nfn[int(sh)])
            out.extend(range(s0, s0 + n))
    return out


def _select_secondary_atoms_by_overlap(
    candidate_atoms: list[int],
    *,
    primary_ao_idx: list[int],
    S_scf: np.ndarray,
    sec_overlap_thr: float,
) -> list[int]:
    if len(candidate_atoms) == 0:
        return []
    if len(primary_ao_idx) == 0:
        return []
    S = np.asarray(S_scf, dtype=np.float64)
    prim = np.asarray(primary_ao_idx, dtype=np.int32)
    thr = float(sec_overlap_thr)
    keep: list[int] = []
    for ia in candidate_atoms:
        # We only know candidate atom here; the caller will translate to AO indices.
        # Filter is applied at atom granularity: keep all AOs on atom if any overlaps.
        # The per-AO version is possible but requires partial-shell handling.
        ia = int(ia)
        keep.append(ia)  # placeholder; actual overlap is checked by caller with AO list
    # The caller will do the overlap check with actual AO indices per atom.
    # Keep list is returned unchanged here to avoid duplicating AO mapping logic.
    return keep


def _build_Z_from_aux_metric(
    cp,
    X_aux_p,
    L,
    *,
    solve_method: str,
    solve_rcond: float = 1e-12,
    store_Z: bool = True,
):
    """Return (Y,Z) for a local-THC block, where Z = Y @ Y.T.

    Y has shape (npt, naux) for the block's auxiliary basis, and Z has shape
    (npt, npt) when `store_Z=True`. When `store_Z=False`, returns Z=None and
    downstream code should apply Z via Y (e.g., n = Y @ (Y.T @ m)).
    """

    solve_s = str(solve_method).strip().lower()
    inv_metric_methods = {"inv_metric", "inv", "metric_inv", "vinv", "v_inv"}
    fit_metric_qr_methods = {"fit_metric_qr", "fit_metric", "qr", "lq", "lstsq"}
    fit_metric_gram_methods = {"fit_metric_gram", "gram"}
    rcond = float(solve_rcond)
    store_Z = bool(store_Z)
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("solve_rcond must be finite and > 0")

    npt_used, naux = map(int, X_aux_p.shape)
    if (solve_s in fit_metric_qr_methods or solve_s in fit_metric_gram_methods) and npt_used < naux:
        raise ValueError(
            f"local-THC block has npt={int(npt_used)} points but aux basis has naux={int(naux)} functions. "
            "Need npt >= naux for the fit_metric solve; increase thc_npt and/or grid size."
        )

    if solve_s in inv_metric_methods:
        try:
            import cupyx.scipy.linalg as cpx_linalg  # noqa: PLC0415

            Xw_T = cpx_linalg.solve_triangular(L, X_aux_p.T, lower=True)
        except Exception:
            Xw_T = cp.linalg.solve(L, X_aux_p.T)
        Xw_p = cp.ascontiguousarray(Xw_T.T)
        Y = Xw_p
        if store_Z:
            Z = cp.ascontiguousarray(Y @ Y.T)
            Z = 0.5 * (Z + Z.T)
        else:
            Z = None
        return Y, Z

    if solve_s in fit_metric_qr_methods:
        # Kept for research; can be ill-conditioned.
        Q, R = cp.linalg.qr(X_aux_p, mode="reduced")
        # Solve R.T @ G = L with SVD regularization on R (robust for ill-conditioned fits).
        try:
            U, s, Vh = cp.linalg.svd(R, full_matrices=False)
            smax = float(cp.max(s).item()) if int(s.size) else 0.0
            if smax == 0.0:
                inv_s = cp.zeros_like(s)
            else:
                cutoff = float(rcond) * float(smax)
                inv_s = cp.where(s > cutoff, 1.0 / s, 0.0)
            RinvT = U @ (inv_s[:, None] * Vh)
            G = RinvT @ L
            Y = cp.ascontiguousarray(Q @ G)
            del U, s, Vh, inv_s, RinvT, G
        except Exception:
            G = cp.linalg.solve(R.T, L)
            Y = cp.ascontiguousarray(Q @ G)
            del G
        if store_Z:
            Z = cp.ascontiguousarray(Y @ Y.T)
            Z = 0.5 * (Z + Z.T)
            # Same guard as global THC: prevent silent NaN/inf explosions.
            try:
                z_finite = bool(cp.all(cp.isfinite(Z)).item())
            except Exception:  # pragma: no cover
                z_finite = True
            if not bool(z_finite):
                raise ValueError(
                    "fit_metric produced non-finite Z entries. "
                    "This usually indicates an ill-conditioned THC grid; try grid_kind='rdvr' with angular_prune=True "
                    "and/or increase thc_npt / angular_n."
                )
            z_max = float(cp.max(cp.abs(Z)).item()) if int(Z.size) else 0.0
            if not np.isfinite(z_max) or z_max > 1e12:
                raise ValueError(
                    f"fit_metric produced huge Z values (max|Z|={z_max:.3e}). "
                    "This usually indicates an ill-conditioned THC grid; try grid_kind='rdvr' with angular_prune=True "
                    "and/or increase thc_npt / angular_n."
                )
        else:
            Z = None
            # Conservative guard without materializing Z: |Z_PQ| <= max_P ||Y_P||^2.
            try:
                y_finite = bool(cp.all(cp.isfinite(Y)).item())
            except Exception:  # pragma: no cover
                y_finite = True
            if not bool(y_finite):
                raise ValueError(
                    "fit_metric produced non-finite Y entries. "
                    "This usually indicates an ill-conditioned THC grid; try grid_kind='rdvr' with angular_prune=True "
                    "and/or increase thc_npt / angular_n."
                )
            max_row_norm_sq = float(cp.max(cp.sum((Y * Y), axis=1)).item()) if int(Y.size) else 0.0
            if not np.isfinite(max_row_norm_sq) or max_row_norm_sq > 1e12:
                raise ValueError(
                    f"fit_metric produced huge Y row norms (max||Y_row||^2={max_row_norm_sq:.3e}). "
                    "This usually indicates an ill-conditioned THC grid; try grid_kind='rdvr' with angular_prune=True "
                    "and/or increase thc_npt / angular_n."
                )
        return Y, Z

    if solve_s in fit_metric_gram_methods:
        Gm = X_aux_p.T @ X_aux_p  # (naux,naux)
        try:
            smax = float(cp.linalg.norm(Gm, ord=2).item())
        except Exception:
            smax = 0.0
        lam = (rcond**2) * max(smax, 1.0)
        if lam != 0.0:
            Gm = Gm + lam * cp.eye(int(Gm.shape[0]), dtype=cp.float64)
        G = cp.linalg.solve(Gm, L)
        Y = cp.ascontiguousarray(X_aux_p @ G)
        if store_Z:
            Z = cp.ascontiguousarray(Y @ Y.T)
            Z = 0.5 * (Z + Z.T)
        else:
            Z = None
        return Y, Z

    raise ValueError("unsupported solve_method for local-THC Z build")


def build_local_thc_factors(
    mol,
    ao_basis,
    aux_basis,
    *,
    S_scf: np.ndarray,
    grid_kind: str = "becke",
    dvr_basis: Any | None = None,
    grid_options: dict[str, Any] | None = None,
    grid_spec: Any | None = None,
    thc_npt: int | None = None,
    config: LocalTHCConfig | None = None,
    metric_backend: str = "gpu_rys",
    metric_mode: str = "warp",
    metric_threads: int = 256,
    solve_method: str = "fit_metric_qr",
    store_Z: bool = True,
    stream: Any = None,
    profile: dict | None = None,
) -> LocalTHCFactors:
    """Build local-THC factors (per-block X/Z) on GPU."""

    cp = _require_cupy()

    from asuka.orbitals.eval_basis_device import (  # noqa: PLC0415
        eval_aos_cart_value_on_points_device,
    )
    from asuka.cueri import df as cueri_df  # noqa: PLC0415

    from asuka.density.grids_device import DeviceGridSpec  # noqa: PLC0415

    grid_spec = DeviceGridSpec() if grid_spec is None else grid_spec
    cfg = LocalTHCConfig() if config is None else config

    grid_kind_s = str(grid_kind).strip().lower()
    grid_opts = {} if grid_options is None else dict(grid_options)

    ao_rep = "cart" if bool(getattr(mol, "cart", False)) else "sph"

    # Map shells to atoms (AO and AUX bases).
    coords = np.asarray(getattr(mol, "coords_bohr", None) if hasattr(mol, "coords_bohr") else mol.coords_bohr, dtype=np.float64)
    shell_to_atom_ao, atom_to_shells_ao = map_shells_to_atoms(np.asarray(ao_basis.shell_cxyz), coords)
    shell_to_atom_aux, atom_to_shells_aux = map_shells_to_atoms(np.asarray(aux_basis.shell_cxyz), coords)
    natm = int(coords.shape[0])

    # Atom AO counts in SCF representation (cart/sph).
    atom_ao_counts = _ao_count_per_atom(np.asarray(ao_basis.shell_l), atom_to_shells_ao, ao_rep=ao_rep)

    # Atom blocking.
    blocks: AtomBlocks = build_atom_blocks(coords, atom_ao_counts, block_max_ao=int(cfg.block_max_ao))

    # Neighbor atoms via Schwarz bounds.
    neigh = build_atom_neighbors_by_schwarz(
        ao_basis,
        shell_to_atom_ao,
        natm=natm,
        aux_schwarz_thr=float(cfg.aux_schwarz_thr),
        threads=int(getattr(grid_spec, "threads", 256)),
        mode=str(metric_mode),
    )

    # Precompute per-atom grids (device arrays).
    pts_atoms, w_atoms = _make_atom_grids_device(
        mol,
        grid_kind=str(grid_kind),
        dvr_basis=dvr_basis,
        grid_options=grid_options,
        grid_spec=grid_spec,
        stream=stream,
    )
    if len(pts_atoms) != natm:
        raise RuntimeError("unexpected per-atom grid count (did mol change?)")

    # Full-shell AO start offsets for the SCF AO representation, to build global AO index lists.
    shell_start_scf, shell_nfn_scf, nao_scf = _shell_ao_starts_scf(ao_basis, ao_rep=ao_rep)
    if profile is not None:
        prof = profile.setdefault("local_thc", {})
        prof["natm"] = int(natm)
        prof["nblocks"] = int(len(blocks.blocks))
        prof["nao"] = int(nao_scf)

    S = np.asarray(S_scf, dtype=np.float64)
    if S.shape != (int(nao_scf), int(nao_scf)):
        raise ValueError("S_scf has wrong shape for AO representation")

    z_dtype_s = str(cfg.z_dtype).strip().lower()
    if z_dtype_s not in {"float64", "float32"}:
        raise ValueError("LocalTHCConfig.z_dtype must be 'float64' or 'float32'")
    z_dtype = cp.float64 if z_dtype_s == "float64" else cp.float32

    solve_s = str(solve_method).strip().lower()
    fit_metric_methods = {
        "fit_metric_qr",
        "fit_metric",
        "qr",
        "lq",
        "lstsq",
        "fit_metric_gram",
        "gram",
    }

    # Optional point selection to improve conditioning of fit-metric Z builds.
    # Weight-only downselect can accidentally drop important near-nucleus points
    # and make X_aux rank-deficient / ill-conditioned.
    point_select = str(
        grid_opts.get(f"{grid_kind_s}_point_select", grid_opts.get("point_select", "auto"))
    ).strip().lower()
    use_pivot = bool(
        solve_s in fit_metric_methods
        and point_select in {"auto", "pivot_qr", "qr_pivot", "rrqr", "pivot", "pivoted_qr"}
    )
    pivot_cache: dict[int, np.ndarray] = {}  # atom -> pivoted point order (len=npt_atom)
    wsum_cache: dict[int, float] = {}  # atom -> sum(weights) (for renormalization)
    aux_basis_atom_cache: dict[int, Any] = {}  # atom -> aux basis subset (cart)
    naux_atom_cache: dict[int, int] = {}  # atom -> naux (cart)

    blocks_out: list[LocalTHCBlock] = []
    downselected_any = False

    for bid, block_atoms in enumerate(blocks.blocks):
        # Auxiliary atoms for the fitting region.
        aux_atoms = union_neighbors(block_atoms, neigh)

        # Primary AO indices (SCF rep) for overlap filtering.
        primary_ao_idx = _atoms_to_ao_indices(
            list(block_atoms),
            atom_to_shells=atom_to_shells_ao,
            shell_start=shell_start_scf,
            shell_nfn=shell_nfn_scf,
        )

        # Candidate secondary atoms: in aux region, excluding this block's atoms.
        block_atom_set = set(int(a) for a in block_atoms)
        cand_atoms = [int(a) for a in aux_atoms if int(a) not in block_atom_set]
        cand_atoms.sort(key=lambda a: (int(blocks.atom_to_block[int(a)]), int(a)))

        # Secondary atoms are chosen by overlap with the primary AO set (atom-level keep).
        sec_atoms_early: list[int] = []
        sec_atoms_late: list[int] = []
        thr = float(cfg.sec_overlap_thr)
        if len(cand_atoms) and len(primary_ao_idx):
            prim = np.asarray(primary_ao_idx, dtype=np.int32)
            for a in cand_atoms:
                ao_a = _atoms_to_ao_indices(
                    [int(a)],
                    atom_to_shells=atom_to_shells_ao,
                    shell_start=shell_start_scf,
                    shell_nfn=shell_nfn_scf,
                )
                if len(ao_a) == 0:
                    continue
                aidx = np.asarray(ao_a, dtype=np.int32)
                # Keep atom if any AO overlaps with the primary set.
                try:
                    vmax = float(np.max(np.abs(S[np.ix_(prim, aidx)])))
                except Exception:
                    vmax = 0.0
                if vmax > thr:
                    ablk = int(blocks.atom_to_block[int(a)])
                    if int(ablk) < int(bid):
                        sec_atoms_early.append(int(a))
                    elif int(ablk) > int(bid):
                        sec_atoms_late.append(int(a))
                    else:
                        # Should be excluded by block_atom_set, but keep robust.
                        pass

        sec_atoms_early.sort(key=lambda a: (int(blocks.atom_to_block[int(a)]), int(a)))
        sec_atoms_late.sort(key=lambda a: (int(blocks.atom_to_block[int(a)]), int(a)))

        # Atoms included in AO basis for this block:
        # [early secondary][primary][late secondary].
        ao_atoms = list(map(int, sec_atoms_early)) + list(map(int, block_atoms)) + list(map(int, sec_atoms_late))

        # AO shells in the desired order.
        ao_shells = _shells_for_atoms(atom_to_shells_ao, ao_atoms)
        ao_basis_blk = subset_cart_basis_by_shells(ao_basis, ao_shells)

        # Global AO indices in the same order as the block AO basis (SCF rep).
        ao_idx_global: list[int] = []
        for sh in ao_shells:
            s0 = int(shell_start_scf[int(sh)])
            n = int(shell_nfn_scf[int(sh)])
            ao_idx_global.extend(range(s0, s0 + n))
        ao_idx_global_np = np.asarray(ao_idx_global, dtype=np.int32)

        # Count early + primary AOs (SCF rep) based on shell sizes.
        early_shells = _shells_for_atoms(atom_to_shells_ao, list(map(int, sec_atoms_early)))
        n_early = 0
        for sh in early_shells:
            n_early += int(shell_nfn_scf[int(sh)])
        n_early = int(n_early)

        # Count primary AOs (SCF rep).
        primary_shells = _shells_for_atoms(atom_to_shells_ao, list(map(int, block_atoms)))
        n_primary = 0
        for sh in primary_shells:
            n_primary += int(shell_nfn_scf[int(sh)])
        n_primary = int(n_primary)

        # Aux basis subset: union of aux shells on auxiliary atoms.
        aux_atoms_sorted = sorted(set(int(a) for a in aux_atoms))
        aux_shells = _shells_for_atoms(atom_to_shells_aux, aux_atoms_sorted)
        aux_basis_blk = subset_cart_basis_by_shells(aux_basis, aux_shells)

        # Block grid points from auxiliary atoms (device arrays).
        aux_atoms = list(map(int, aux_atoms_sorted))
        pts_list = [pts_atoms[int(a)] for a in aux_atoms]
        w_list = [w_atoms[int(a)] for a in aux_atoms]
        a_list = [cp.full((int(wi.shape[0]),), int(a), dtype=cp.int32) for a, wi in zip(aux_atoms, w_list)]
        npt_avail = int(sum(int(wi.shape[0]) for wi in w_list))
        if npt_avail <= 0:
            raise RuntimeError("local-THC block has no grid points (check prune_tol / grid spec)")

        # Always compute naux_total so npt scales with aux basis size (not n_primary).
        # For a single-block system (all atoms in one block), n_primary << naux,
        # so using npt_factor * n_primary severely under-samples the grid.
        from asuka.cueri.cart import ncart  # noqa: PLC0415

        naux_atom = np.zeros((len(aux_atoms),), dtype=np.int64)
        for i, a in enumerate(aux_atoms):
            if int(a) not in naux_atom_cache:
                shells = atom_to_shells_aux[int(a)]
                bas_i = subset_cart_basis_by_shells(aux_basis, shells)
                aux_basis_atom_cache[int(a)] = bas_i
                shell_l = np.asarray(bas_i.shell_l, dtype=np.int32).ravel()
                shell_start = np.asarray(bas_i.shell_ao_start, dtype=np.int32).ravel()
                if int(shell_l.size):
                    nfn = np.asarray([ncart(int(l)) for l in shell_l.tolist()], dtype=np.int32)
                    naux_i = int(np.max(shell_start + nfn))
                else:
                    naux_i = 0
                naux_atom_cache[int(a)] = int(naux_i)
            naux_atom[int(i)] = int(naux_atom_cache[int(a)])
        naux_total = int(np.sum(naux_atom))

        if bool(getattr(cfg, "no_point_downselect", False)):
            npt_target = int(npt_avail)
        elif thc_npt is not None:
            npt_target = int(max(1, int(thc_npt)))
        else:
            # Scale with naux_total (aux basis size) for accuracy; n_primary
            # (AO count) is often much smaller, leading to insufficient sampling.
            npt_target = int(cfg.npt_factor) * int(max(1, int(naux_total) if int(naux_total) > 0 else int(n_primary)))
            npt_target = max(int(cfg.npt_min), min(int(cfg.npt_max), int(npt_target)))

        if solve_s in fit_metric_methods:
            if int(naux_total) > 0:
                if int(npt_avail) < int(naux_total):
                    raise ValueError(
                        f"local-THC block has npt_avail={int(npt_avail)} but needs at least naux={int(naux_total)} points "
                        f"for solve_method={solve_s!r}. Increase grid size / thc_npt, or use solve_method='inv_metric'."
                    )
                npt_target = max(int(npt_target), int(naux_total))
        npt_target = min(int(npt_target), int(npt_avail))
        downselected = (not bool(getattr(cfg, "no_point_downselect", False))) and (int(npt_target) < int(npt_avail))
        downselected_any = bool(downselected_any) or bool(downselected)

        # Downselect points.
        #
        # If requested, use per-atom RRQR point selection (CPU) on the local aux
        # basis to improve conditioning of the fit-metric solve.
        pts = None
        w = None
        p_atom = None

        if bool(getattr(cfg, "no_point_downselect", False)):
            pts = cp.ascontiguousarray(cp.concatenate(pts_list, axis=0))
            w = cp.ascontiguousarray(cp.concatenate(w_list, axis=0)).ravel()
            p_atom = cp.ascontiguousarray(cp.concatenate(a_list, axis=0)).ravel()
        elif bool(use_pivot) and solve_s in fit_metric_methods and int(npt_target) < int(npt_avail):
            try:
                import scipy.linalg as sp_linalg  # noqa: PLC0415

                from asuka.orbitals.eval_cart import eval_basis_cart_value_on_points  # noqa: PLC0415

                # Reuse per-atom aux sizes from above.
                naux_atom = np.asarray([int(naux_atom_cache.get(int(a), 0)) for a in aux_atoms], dtype=np.int64)
                naux_total = int(np.sum(naux_atom))
                if naux_total <= 0:
                    raise RuntimeError("pivot selection found no auxiliary functions")

                avail = np.asarray([int(wi.shape[0]) for wi in w_list], dtype=np.int64)
                npt_req = int(npt_target)
                if npt_req < naux_total:
                    raise ValueError(f"Need npt >= naux for fit_metric. Got npt={npt_req} but naux={naux_total}.")

                # Allocate points per atom proportional to local aux size, with k_i >= naux_i.
                frac = (float(npt_req) * naux_atom.astype(np.float64)) / float(naux_total)
                alloc = np.floor(frac).astype(np.int64)
                alloc = np.maximum(alloc, naux_atom)
                alloc = np.minimum(alloc, avail)

                diff = int(npt_req - int(np.sum(alloc)))
                frac_part = frac - np.floor(frac)
                if diff > 0:
                    order = np.argsort(-frac_part, kind="stable")
                    for idx in order.tolist():
                        if diff == 0:
                            break
                        if int(alloc[idx]) < int(avail[idx]):
                            alloc[idx] += 1
                            diff -= 1
                elif diff < 0:
                    order = np.argsort(frac_part, kind="stable")
                    for idx in order.tolist():
                        while diff < 0 and int(alloc[idx]) > int(naux_atom[idx]):
                            alloc[idx] -= 1
                            diff += 1

                pts_sel: list[Any] = []
                w_sel: list[Any] = []
                a_sel: list[Any] = []
                for a, pts_i, w_i, k in zip(aux_atoms, pts_list, w_list, alloc.tolist()):
                    k = int(k)
                    if k <= 0:
                        continue
                    a = int(a)
                    if int(w_i.shape[0]) == 0:
                        continue

                    piv = pivot_cache.get(a)
                    if piv is None:
                        bas_i = aux_basis_atom_cache.get(a)
                        if bas_i is None:
                            shells = atom_to_shells_aux[int(a)]
                            bas_i = subset_cart_basis_by_shells(aux_basis, shells)
                            aux_basis_atom_cache[int(a)] = bas_i
                        pts_np = np.ascontiguousarray(cp.asnumpy(pts_i), dtype=np.float64).reshape((-1, 3))
                        w_np = np.ascontiguousarray(cp.asnumpy(w_i), dtype=np.float64).ravel()
                        phi = eval_basis_cart_value_on_points(bas_i, pts_np)
                        Xw = phi * np.sqrt(w_np)[:, None]
                        A = Xw.T  # (naux_i, npt_i)
                        _Q, _R, piv = sp_linalg.qr(A, pivoting=True, mode="economic")
                        piv = np.asarray(piv, dtype=np.int64)
                        pivot_cache[a] = piv
                        wsum_cache[a] = float(np.sum(w_np))

                    idx_np = np.asarray(piv[:k], dtype=np.int64)
                    idx = cp.asarray(idx_np, dtype=cp.int64)
                    pts_j = cp.ascontiguousarray(pts_i[idx])
                    w_j = cp.ascontiguousarray(w_i[idx])

                    # Renormalize per-atom weights to preserve total weight.
                    w_sum_all = float(wsum_cache.get(a, float(cp.sum(w_i).item())))
                    w_sum_sel = float(cp.sum(w_j).item())
                    if w_sum_sel != 0.0:
                        w_j = w_j * (w_sum_all / w_sum_sel)
                    pts_sel.append(pts_j)
                    w_sel.append(w_j)
                    a_sel.append(cp.full((int(w_j.shape[0]),), int(a), dtype=cp.int32))

                if len(pts_sel) == 0:
                    raise RuntimeError("pivot selection produced no grid points")
                pts = cp.ascontiguousarray(cp.concatenate(pts_sel, axis=0))
                w = cp.ascontiguousarray(cp.concatenate(w_sel, axis=0)).ravel()
                p_atom = cp.ascontiguousarray(cp.concatenate(a_sel, axis=0)).ravel()
            except Exception:
                pts = None
                w = None
                p_atom = None

        if pts is None or w is None or p_atom is None:
            # Fallback: weight-based downselect from all available points.
            pts_all = cp.concatenate(pts_list, axis=0) if pts_list else cp.zeros((0, 3), dtype=cp.float64)
            w_all = cp.concatenate(w_list, axis=0) if w_list else cp.zeros((0,), dtype=cp.float64)
            a_all = cp.concatenate(a_list, axis=0) if a_list else cp.zeros((0,), dtype=cp.int32)
            pts, w, idx = _downselect_by_weight(cp, pts_all, w_all, npt=int(npt_target), return_idx=True)
            p_atom = a_all[idx]
            del pts_all, w_all

        # AO collocation: X_cart = w^(1/4) * phi_cart(r)
        ao_cart = eval_aos_cart_value_on_points_device(
            ao_basis_blk,
            pts,
            threads=int(getattr(grid_spec, "threads", 256)),
            stream=stream,
            sync=True,
        )
        w_quart = cp.sqrt(cp.sqrt(w))
        X_cart = cp.ascontiguousarray(ao_cart * w_quart[:, None])
        del ao_cart

        # Optional cart->sph transform for this subset basis.
        if ao_rep == "sph":
            from asuka.integrals.cart2sph import build_cart2sph_matrix, compute_sph_layout_from_cart_basis  # noqa: PLC0415

            shell_l_blk = np.asarray(ao_basis_blk.shell_l, dtype=np.int32).ravel()
            shell_start_cart_blk = np.asarray(ao_basis_blk.shell_ao_start, dtype=np.int32).ravel()
            shell_start_sph_blk, nao_sph_blk = compute_sph_layout_from_cart_basis(ao_basis_blk)
            nao_cart_blk = int(X_cart.shape[1])
            T = build_cart2sph_matrix(shell_l_blk, shell_start_cart_blk, shell_start_sph_blk, nao_cart_blk, int(nao_sph_blk))
            T_dev = cp.asarray(T, dtype=cp.float64)
            X = cp.ascontiguousarray(X_cart @ T_dev)
            del T_dev, T
        else:
            X = X_cart

        # Auxiliary collocation: X_aux = w^(1/2) * chi_cart(r)
        aux_cart = eval_aos_cart_value_on_points_device(
            aux_basis_blk,
            pts,
            threads=int(getattr(grid_spec, "threads", 256)),
            stream=stream,
            sync=True,
        )
        w_sqrt = cp.sqrt(w)
        X_aux_p = cp.ascontiguousarray(aux_cart * w_sqrt[:, None])
        del aux_cart

        # Aux metric + Cholesky
        V = cueri_df.metric_2c2e_basis(
            aux_basis_blk,
            stream=stream,
            backend=str(metric_backend),
            mode=str(metric_mode),
            threads=int(metric_threads),
        )
        L = cueri_df.cholesky_metric(V)

        Y, Z = _build_Z_from_aux_metric(cp, X_aux_p, L, solve_method=str(solve_method), store_Z=bool(store_Z))
        Y = cp.ascontiguousarray(cp.asarray(Y, dtype=z_dtype))
        if Z is not None:
            Z = cp.ascontiguousarray(cp.asarray(Z, dtype=z_dtype))

        # Sanity: X columns must match local AO count.
        if int(X.shape[1]) != int(ao_idx_global_np.size):
            raise RuntimeError("local-THC X column count mismatch with ao_idx_global")

        blk_meta: dict[str, Any] = {
            "grid_kind": str(grid_kind_s),
            "becke_n": int(getattr(grid_spec, "becke_n", 3)),
            "prune_tol": float(getattr(grid_spec, "prune_tol", 1e-16)),
            "grid_options": dict(grid_opts),
            "solve_method": str(solve_method),
            "downselected": bool(downselected),
            "point_atom": cp.ascontiguousarray(cp.asarray(p_atom, dtype=cp.int32).ravel()),
            "ao_shells": tuple(int(s) for s in ao_shells),
            "aux_shells": tuple(int(s) for s in aux_shells),
        }

        blocks_out.append(
            LocalTHCBlock(
                block_id=int(bid),
                ao_idx_global=np.ascontiguousarray(ao_idx_global_np),
                n_early=int(n_early),
                n_primary=int(n_primary),
                atoms_primary=tuple(int(a) for a in block_atoms),
                atoms_secondary_early=tuple(int(a) for a in sec_atoms_early),
                atoms_secondary_late=tuple(int(a) for a in sec_atoms_late),
                atoms_aux=tuple(int(a) for a in aux_atoms_sorted),
                X=X,
                Y=Y,
                Z=Z,
                points=pts,
                weights=w,
                L_metric=L,
                meta=blk_meta,
            )
        )

    meta: dict[str, Any] = {
        "grid_kind": str(grid_kind_s),
        "becke_n": int(getattr(grid_spec, "becke_n", 3)),
        "prune_tol": float(getattr(grid_spec, "prune_tol", 1e-16)),
        "grid_options": dict(grid_opts),
        "solve_method": str(solve_method),
        "downselected": bool(downselected_any),
    }

    return LocalTHCFactors(
        blocks=tuple(blocks_out),
        nao=int(nao_scf),
        ao_rep=str(ao_rep),
        L_metric_full=None,
        meta=meta,
    )


__all__ = ["LocalTHCBlock", "LocalTHCFactors", "build_local_thc_factors"]
