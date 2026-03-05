from __future__ import annotations

"""THC factor construction (GPU-first).

Build THC factors (X, Z) suitable for O(N^3) J/K contractions:

  (mu nu | la si) ~= sum_{P,Q} X[P,mu] X[P,nu] Z[P,Q] X[Q,la] X[Q,si]

We use:
- AO collocation: X[P,mu] = w_P^(1/4) * phi_mu(r_P)
- Aux collocation: X_aux[P,A] = w_P^(1/2) * chi_A(r_P)
- Auxiliary Coulomb metric (A|B) = V = L L^T from cuERI
- Central metric in point basis via auxiliary-metric fit (default; paper-consistent):
    Solve (X_aux^T) @ Y = L (underdetermined; requires npt >= naux),
    then Z = Y Y^T.

A stable alternative is the auxiliary-metric inverse projection:
    Z = X_aux V^{-1} X_aux^T = (X_aux L^{-T}) (X_aux L^{-T})^T.

All heavy tensors live on the GPU as CuPy arrays.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import time


@dataclass(frozen=True)
class THCFactors:
    """THC factors stored on the GPU.

    Attributes
    ----------
    X : cupy.ndarray
        Weighted AO collocation matrix with shape (npt, nao) (points-major).
    Y : cupy.ndarray
        Factor of the central metric with shape (npt, naux) such that
        ``Z = Y @ Y.T`` (up to numerical symmetrization). Here ``naux`` is
        the size of the auxiliary basis used in the metric build.
    Z : cupy.ndarray
        Central metric with shape (npt, npt), symmetric.
    points : cupy.ndarray
        Grid points (npt,3) in Bohr.
    weights : cupy.ndarray
        Grid weights (npt,) (quadrature).
    L_metric : cupy.ndarray
        Cholesky factor of the auxiliary Coulomb metric (naux,naux), lower.
        Useful for reusing in DF warmup/reference builds.
    """

    X: Any
    Y: Any
    Z: Any
    points: Any
    weights: Any
    L_metric: Any
    meta: dict[str, Any] | None = None


def _require_cupy():
    try:
        import cupy as cp  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("THC factor construction requires CuPy") from e
    return cp


def _profile_ms_set(d: dict, key: str, ms: float) -> None:
    try:
        d[key] = float(ms)
    except Exception:  # pragma: no cover
        d[key] = ms


def _profile_ms_add(d: dict, key: str, ms: float) -> None:
    try:
        d[key] = float(d.get(key, 0.0)) + float(ms)
    except Exception:  # pragma: no cover
        d[key] = d.get(key, 0.0) + ms


def _downselect_by_weight(cp, points, weights, *, npt: int):
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
        return cp.ascontiguousarray(points), cp.ascontiguousarray(weights)

    # Select the top-npt weights (deterministic tie-break by original index).
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

    # Renormalize weights to preserve total quadrature weight.
    w_sum = cp.sum(w)
    if float(w_sum) != 0.0:
        w = w * (w_all_sum / w_sum)
    return cp.ascontiguousarray(pts), cp.ascontiguousarray(w)


def _solve_Y_qr(cp, X_aux_p, L):
    # X_aux_p: (npt,naux) == (X_aux.T) in the notes.
    Q, R = cp.linalg.qr(X_aux_p, mode="reduced")  # Q:(npt,naux), R:(naux,naux)
    # Solve R.T @ G = L (R is upper -> R.T lower)
    G = cp.linalg.solve(R.T, L)
    return cp.ascontiguousarray(Q @ G)  # (npt,naux)


def _solve_Y_gram(cp, X_aux_p, L):
    # Minimum-norm solution: Y = X_aux^T (X_aux X_aux^T)^{-1} L.
    Gm = X_aux_p.T @ X_aux_p  # (naux,naux)
    G = cp.linalg.solve(Gm, L)  # (naux,naux)
    return cp.ascontiguousarray(X_aux_p @ G)  # (npt,naux)


def _solve_Y_lstsq(cp, X_aux_p, L, *, rcond: float):
    # SVD-based least-squares with cutoff; returns min-norm solution for the
    # underdetermined system X_aux_p.T @ Y = L.
    #
    # Shapes:
    #   A = X_aux_p.T: (naux, npt) (wide)
    #   B = L:         (naux, naux)
    #   Y:             (npt, naux)
    Y, _res, _rank, _s = cp.linalg.lstsq(X_aux_p.T, L, rcond=float(rcond))
    return cp.ascontiguousarray(Y)

def build_thc_factors(
    mol_or_coords: Any,
    ao_basis: Any,
    aux_basis: Any,
    *,
    sph_map: Any | None = None,
    grid_spec: Any | None = None,
    grid_kind: str = "becke",
    dvr_basis: Any | None = None,
    grid_options: dict[str, Any] | None = None,
    npt: int | None = None,
    # Aux-metric params (cuERI)
    metric_backend: str = "gpu_rys",
    metric_mode: str = "warp",
    metric_threads: int = 256,
    # Linear solve for Y
    solve_method: str = "fit_metric_qr",
    solve_rcond: float | None = 1e-12,
    # GPU execution
    stream: Any = None,
    profile: dict | None = None,
) -> THCFactors:
    """Build THC factors (X,Z) on GPU.

    Parameters
    ----------
    mol_or_coords
        Molecule-like object or coordinates accepted by `asuka.density` grids.
    ao_basis, aux_basis
        Packed cartesian AO and auxiliary bases (cart basis objects).
    sph_map
        Optional spherical AO transform metadata (from `frontend.scf._apply_sph_transform`).
        If provided, X is returned in spherical AO representation.
    grid_spec
        `asuka.density.DeviceGridSpec` or None (defaults to DeviceGridSpec()).
    npt
        If provided and smaller than the full Becke grid, downselect to this
        many points by largest weights and renormalize.
    """

    cp = _require_cupy()
    from asuka.cueri import df as cueri_df  # noqa: PLC0415
    from asuka.orbitals.eval_basis_device import (  # noqa: PLC0415
        eval_aos_cart_value_on_points_device,
    )
    from asuka.density import DeviceGridSpec, make_becke_grid_device  # noqa: PLC0415

    grid_spec = DeviceGridSpec() if grid_spec is None else grid_spec

    grid_kind_s = str(grid_kind).strip().lower()
    grid_opts = {} if grid_options is None else dict(grid_options)
    grid_npt_full_override: int | None = None
    solve_s_pre = str(solve_method).strip().lower()
    fit_metric_methods_pre = {"fit_metric_qr", "fit_metric", "qr", "lq", "lstsq", "fit_metric_gram", "gram"}

    prof = profile.setdefault("thc_factors", {}) if profile is not None else None
    t_total = time.perf_counter() if prof is not None else None

    def _sync_if_profile() -> None:
        if prof is None:
            return
        try:
            cp.cuda.runtime.deviceSynchronize()
        except Exception:  # pragma: no cover
            # Best-effort profiling only.
            pass

    t_grid = time.perf_counter() if prof is not None else None
    if grid_kind_s in {"becke", "becke_grid", "atom", "atom-centered"}:
        # If the caller requests a reduced point count, avoid a global
        # weight-based downselect that can accidentally starve some atoms of
        # grid coverage for larger molecules. Instead, downselect per-atom
        # (using the Becke atom ordering) and then concatenate.
        if npt is None:
            pts_all, w_all = make_becke_grid_device(
                mol_or_coords,
                radial_n=int(getattr(grid_spec, "radial_n", 50)),
                angular_n=int(getattr(grid_spec, "angular_n", 302)),
                angular_kind=str(getattr(grid_spec, "angular_kind", "auto")),
                rmax=float(getattr(grid_spec, "rmax", 20.0)),
                becke_n=int(getattr(grid_spec, "becke_n", 3)),
                prune_tol=float(getattr(grid_spec, "prune_tol", 1e-16)),
                threads=int(getattr(grid_spec, "threads", 256)),
                stream=stream,
            )
        else:
            from asuka.density.grids_device import iter_becke_grid_device  # noqa: PLC0415

            pts_atoms: list[Any] = []
            w_atoms: list[Any] = []
            for pts_i, w_i in iter_becke_grid_device(
                mol_or_coords,
                radial_n=int(getattr(grid_spec, "radial_n", 50)),
                angular_n=int(getattr(grid_spec, "angular_n", 302)),
                angular_kind=str(getattr(grid_spec, "angular_kind", "auto")),
                rmax=float(getattr(grid_spec, "rmax", 20.0)),
                becke_n=int(getattr(grid_spec, "becke_n", 3)),
                block_size=10**9,  # one block per atom
                prune_tol=float(getattr(grid_spec, "prune_tol", 1e-16)),
                threads=int(getattr(grid_spec, "threads", 256)),
                stream=stream,
            ):
                pts_atoms.append(cp.ascontiguousarray(cp.asarray(pts_i, dtype=cp.float64)))
                w_atoms.append(cp.ascontiguousarray(cp.asarray(w_i, dtype=cp.float64).ravel()))

            grid_npt_full_override = int(sum(int(wi.shape[0]) for wi in w_atoms))
            natm = int(len(pts_atoms))
            npt_req = int(npt)
            if natm <= 0:
                pts_all, w_all = cp.zeros((0, 3), dtype=cp.float64), cp.zeros((0,), dtype=cp.float64)
            else:
                if npt_req <= 0:
                    raise ValueError("npt must be > 0")
                base = int(npt_req // natm)
                rem = int(npt_req - base * natm)
                alloc = [base + (1 if i < rem else 0) for i in range(natm)]
                pts_sel: list[Any] = []
                w_sel: list[Any] = []
                for pts_i, w_i, ni in zip(pts_atoms, w_atoms, alloc):
                    if int(ni) <= 0:
                        continue
                    pts_j, w_j = _downselect_by_weight(cp, pts_i, w_i, npt=int(ni))
                    pts_sel.append(pts_j)
                    w_sel.append(w_j)
                if len(pts_sel) == 0:
                    pts_all, w_all = cp.zeros((0, 3), dtype=cp.float64), cp.zeros((0,), dtype=cp.float64)
                else:
                    pts_all = cp.ascontiguousarray(cp.concatenate(pts_sel, axis=0))
                    w_all = cp.ascontiguousarray(cp.concatenate(w_sel, axis=0))

            # Mark as already downselected.
            npt = None
    elif grid_kind_s in {"rdvr", "r-dvr", "r_dvr"}:
        if dvr_basis is None:
            raise ValueError("grid_kind='rdvr' requires dvr_basis (use aux_basis or provide a separate DVR basis)")
        # Prefer the CUDA-backed R-DVR grid generator to keep Becke partitioning
        # on the GPU (paper-faithful / apple-to-apple).
        from asuka.density.dvr_grids_device import (  # noqa: PLC0415
            iter_rdvr_grid_device,
            make_rdvr_grid_device,
        )

        if npt is None:
            pts_all, w_all = make_rdvr_grid_device(
                mol_or_coords,
                dvr_basis,
                angular_n=int(getattr(grid_spec, "angular_n", 302)),
                angular_kind=str(getattr(grid_spec, "angular_kind", "auto")),
                radial_rmax=float(grid_opts.get("radial_rmax", getattr(grid_spec, "rmax", 20.0))),
                becke_n=int(getattr(grid_spec, "becke_n", 3)),
                angular_prune=bool(grid_opts.get("angular_prune", True)),
                prune_tol=float(getattr(grid_spec, "prune_tol", 1e-16)),
                ortho_cutoff=float(grid_opts.get("ortho_cutoff", 1e-10)),
                threads=int(getattr(grid_spec, "threads", 256)),
                stream=stream,
            )
        else:
            # Same rationale as Becke: a global weight-based downselect can starve
            # some atoms of coverage on large molecules, which can make the fit-metric
            # Z build rank-deficient / ill-conditioned. Downselect per-atom first.
            npt_req = int(npt)
            if npt_req <= 0:
                raise ValueError("npt must be > 0")

            point_select_req = str(grid_opts.get("rdvr_point_select", "auto")).strip().lower()
            pivot_keys = {"pivot_qr", "qr_pivot", "rrqr", "pivot", "pivoted_qr"}
            needs_fit_metric = solve_s_pre in fit_metric_methods_pre

            if point_select_req in {"", "auto"}:
                # For fit-metric Z builds, conditioning is critical; default to
                # pivoted-QR selection (paper-style) rather than weight-only.
                point_select = "pivot" if needs_fit_metric else "weight"
            else:
                point_select = point_select_req
            explicit_pivot = point_select_req in pivot_keys

            # Build full per-atom R-DVR grids on the GPU (one block per atom).
            pts_atoms: list[Any] = []
            w_atoms: list[Any] = []
            for pts_i, w_i in iter_rdvr_grid_device(
                mol_or_coords,
                dvr_basis,
                angular_n=int(getattr(grid_spec, "angular_n", 302)),
                angular_kind=str(getattr(grid_spec, "angular_kind", "auto")),
                radial_rmax=float(grid_opts.get("radial_rmax", getattr(grid_spec, "rmax", 20.0))),
                becke_n=int(getattr(grid_spec, "becke_n", 3)),
                block_size=10**9,  # one block per atom
                angular_prune=bool(grid_opts.get("angular_prune", True)),
                prune_tol=float(getattr(grid_spec, "prune_tol", 1e-16)),
                ortho_cutoff=float(grid_opts.get("ortho_cutoff", 1e-10)),
                threads=int(getattr(grid_spec, "threads", 256)),
                stream=stream,
            ):
                pts_atoms.append(cp.ascontiguousarray(cp.asarray(pts_i, dtype=cp.float64)))
                w_atoms.append(cp.ascontiguousarray(cp.asarray(w_i, dtype=cp.float64).ravel()))

            grid_npt_full_override = int(sum(int(wi.size) for wi in w_atoms))
            natm = int(len(pts_atoms))
            if natm <= 0:
                pts_all, w_all = cp.zeros((0, 3), dtype=cp.float64), cp.zeros((0,), dtype=cp.float64)
                npt = None
            elif point_select in pivot_keys:
                # Pivoted QR selection (rank-revealing QR) using GPU aux basis
                # evaluation. Only the QR pivoting itself is done on CPU (SciPy).
                try:
                    import scipy.linalg as sp_linalg  # noqa: PLC0415

                    from asuka.cueri.basis_subset import subset_cart_basis_by_shells  # noqa: PLC0415
                    from asuka.cueri.cart import ncart  # noqa: PLC0415
                    from asuka.hf.local_thc_partition import map_shells_to_atoms  # noqa: PLC0415
                    from asuka.density.grids import _coords_bohr  # noqa: PLC0415

                    coords = np.asarray(_coords_bohr(mol_or_coords), dtype=np.float64).reshape((-1, 3))
                    if int(coords.shape[0]) != int(natm):
                        raise ValueError(
                            f"R-DVR grid atom count mismatch: grid natm={int(natm)} but coords natm={int(coords.shape[0])}"
                        )
                    _sh2a, a2sh = map_shells_to_atoms(np.asarray(aux_basis.shell_cxyz), coords)

                    # Aux-basis function count per atom (cart) + per-atom subsets.
                    naux_atom = np.zeros((natm,), dtype=np.int64)
                    aux_basis_atom: list[Any] = []
                    for ia in range(natm):
                        shells = a2sh[int(ia)]
                        bas_i = subset_cart_basis_by_shells(aux_basis, shells)
                        aux_basis_atom.append(bas_i)
                        shell_l = np.asarray(bas_i.shell_l, dtype=np.int32).ravel()
                        shell_start = np.asarray(bas_i.shell_ao_start, dtype=np.int32).ravel()
                        if int(shell_l.size):
                            nfn = np.asarray([ncart(int(l)) for l in shell_l.tolist()], dtype=np.int32)
                            naux_i = int(np.max(shell_start + nfn))
                        else:
                            naux_i = 0
                        naux_atom[int(ia)] = int(naux_i)

                    naux_total = int(np.sum(naux_atom))
                    if naux_total <= 0:
                        raise RuntimeError("R-DVR pivot selection found no auxiliary functions")
                    if needs_fit_metric and npt_req < naux_total:
                        raise ValueError(
                            f"Need npt >= naux for fit_metric. Got npt={int(npt_req)} but naux={int(naux_total)}."
                        )

                    # Allocate points per atom proportional to local aux size.
                    # For fit-metric solves, also enforce k_i >= naux_i to avoid
                    # local rank deficiency.
                    avail = np.asarray([int(wi.size) for wi in w_atoms], dtype=np.int64)
                    if naux_total > 0:
                        frac = (float(npt_req) * naux_atom.astype(np.float64)) / float(naux_total)
                    else:  # pragma: no cover
                        frac = np.zeros((natm,), dtype=np.float64)
                    alloc = np.floor(frac).astype(np.int64)
                    if needs_fit_metric:
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
                            while diff < 0 and (not needs_fit_metric or int(alloc[idx]) > int(naux_atom[idx])):
                                alloc[idx] -= 1
                                diff += 1

                    pts_sel: list[Any] = []
                    w_sel: list[Any] = []
                    for ia in range(natm):
                        k = int(alloc[int(ia)])
                        if k <= 0:
                            continue
                        pts_i = pts_atoms[int(ia)]
                        w_i = w_atoms[int(ia)]
                        n_i = int(w_i.size)
                        if n_i <= 0:
                            continue
                        if k >= n_i:
                            pts_j = pts_i
                            w_j = w_i
                        else:
                            bas_i = aux_basis_atom[int(ia)]
                            phi = eval_aos_cart_value_on_points_device(
                                bas_i,
                                pts_i,
                                threads=int(getattr(grid_spec, "threads", 256)),
                                stream=stream,
                                sync=True,
                            )  # (npt_i,naux_i)
                            sw = cp.sqrt(w_i)  # (npt_i,)
                            A = phi.T * sw[None, :]  # (naux_i,npt_i)
                            A_h = cp.asnumpy(A)
                            _Q, _R, piv = sp_linalg.qr(A_h, pivoting=True, mode="economic")
                            idx_h = np.asarray(piv[:k], dtype=np.int64)
                            idx = cp.asarray(idx_h, dtype=cp.int64)
                            pts_j = pts_i[idx]
                            w_j = w_i[idx]
                            # Renormalize per-atom weights to preserve total weight.
                            w_sum_all = cp.sum(w_i)
                            w_sum_sel = cp.sum(w_j)
                            if float(w_sum_sel) != 0.0:
                                w_j = w_j * (w_sum_all / w_sum_sel)

                        pts_sel.append(cp.ascontiguousarray(pts_j))
                        w_sel.append(cp.ascontiguousarray(w_j))

                    if len(pts_sel) == 0:
                        pts_all, w_all = cp.zeros((0, 3), dtype=cp.float64), cp.zeros((0,), dtype=cp.float64)
                    else:
                        pts_all = cp.ascontiguousarray(cp.concatenate(pts_sel, axis=0))
                        w_all = cp.ascontiguousarray(cp.concatenate(w_sel, axis=0))
                    npt = None
                except Exception:
                    if explicit_pivot:
                        raise
                    # Fall back to weight-based selection if anything goes wrong.
                    point_select = "weight"

            if npt is not None and point_select not in pivot_keys:
                # Weight-based downselect per atom (GPU).
                base = int(npt_req // natm)
                rem = int(npt_req - base * natm)
                alloc = [base + (1 if i < rem else 0) for i in range(natm)]
                pts_sel = []
                w_sel = []
                for pts_i, w_i, ni in zip(pts_atoms, w_atoms, alloc):
                    if int(ni) <= 0:
                        continue
                    pts_j, w_j = _downselect_by_weight(cp, pts_i, w_i, npt=int(ni))
                    pts_sel.append(pts_j)
                    w_sel.append(w_j)
                if len(pts_sel) == 0:
                    pts_all, w_all = cp.zeros((0, 3), dtype=cp.float64), cp.zeros((0,), dtype=cp.float64)
                else:
                    pts_all = cp.ascontiguousarray(cp.concatenate(pts_sel, axis=0))
                    w_all = cp.ascontiguousarray(cp.concatenate(w_sel, axis=0))

                npt = None
    elif grid_kind_s in {"fdvr", "f-dvr", "f_dvr"}:
        if dvr_basis is None:
            raise ValueError("grid_kind='fdvr' requires dvr_basis (use aux_basis or provide a separate DVR basis)")
        from asuka.density.dvr_grids import make_fdvr_grid_device  # noqa: PLC0415

        pts_all, w_all = make_fdvr_grid_device(
            dvr_basis,
            ortho_cutoff=float(grid_opts.get("ortho_cutoff", 1e-10)),
            max_sweeps=int(grid_opts.get("max_sweeps", 50)),
            tol=float(grid_opts.get("tol", 1e-12)),
            prune_tol=float(getattr(grid_spec, "prune_tol", 1e-16)),
            validate=bool(grid_opts.get("validate", True)),
            overlap_max_abs_tol=float(grid_opts.get("overlap_max_abs_tol", 1e-3)),
        )
    else:
        raise ValueError("grid_kind must be one of: 'becke', 'rdvr', 'fdvr'")
    pts_all = cp.ascontiguousarray(cp.asarray(pts_all, dtype=cp.float64))
    w_all = cp.ascontiguousarray(cp.asarray(w_all, dtype=cp.float64).ravel())

    _sync_if_profile()
    if prof is not None and t_grid is not None:
        _profile_ms_set(prof, "grid_ms", (time.perf_counter() - float(t_grid)) * 1000.0)
        prof["grid_npt_full"] = int(w_all.shape[0]) if grid_npt_full_override is None else int(grid_npt_full_override)

    t_down = time.perf_counter() if prof is not None else None
    if npt is not None:
        pts, w = _downselect_by_weight(cp, pts_all, w_all, npt=int(npt))
    else:
        pts, w = pts_all, w_all

    _sync_if_profile()
    if prof is not None:
        if t_down is not None:
            _profile_ms_set(prof, "downselect_ms", (time.perf_counter() - float(t_down)) * 1000.0)
        prof["grid_npt_used"] = int(w.shape[0])

    solve_s = str(solve_method).strip().lower()
    inv_metric_methods = {"inv_metric", "inv", "metric_inv", "vinv", "v_inv"}
    fit_metric_qr_methods = {"fit_metric_qr", "fit_metric", "qr", "lq", "lstsq"}
    fit_metric_gram_methods = {"fit_metric_gram", "gram"}

    # ---- Collocation ----
    # AO: w^(1/4) * phi(r)
    t_ao = time.perf_counter() if prof is not None else None
    ao_cart = eval_aos_cart_value_on_points_device(ao_basis, pts, threads=int(getattr(grid_spec, "threads", 256)), stream=stream, sync=True)
    w_quart = cp.sqrt(cp.sqrt(w))
    X_cart = cp.ascontiguousarray(ao_cart * w_quart[:, None])
    del ao_cart
    _sync_if_profile()
    if prof is not None and t_ao is not None:
        _profile_ms_set(prof, "x_ao_ms", (time.perf_counter() - float(t_ao)) * 1000.0)

    if sph_map is not None:
        t_sph = time.perf_counter() if prof is not None else None
        from asuka.integrals.cart2sph import coerce_sph_map  # noqa: PLC0415

        T = np.asarray(coerce_sph_map(sph_map).T_c2s, dtype=np.float64)
        T_dev = cp.asarray(T, dtype=cp.float64)
        X = cp.ascontiguousarray(X_cart @ T_dev)
        del T_dev
        _sync_if_profile()
        if prof is not None and t_sph is not None:
            _profile_ms_set(prof, "x_sph_ms", (time.perf_counter() - float(t_sph)) * 1000.0)
    else:
        X = X_cart

    # Aux: w^(1/2) * chi(r)
    t_aux = time.perf_counter() if prof is not None else None
    aux_cart = eval_aos_cart_value_on_points_device(aux_basis, pts, threads=int(getattr(grid_spec, "threads", 256)), stream=stream, sync=True)
    w_sqrt = cp.sqrt(w)
    X_aux_p = cp.ascontiguousarray(aux_cart * w_sqrt[:, None])  # (npt,naux)
    del aux_cart
    _sync_if_profile()
    if prof is not None and t_aux is not None:
        _profile_ms_set(prof, "x_aux_ms", (time.perf_counter() - float(t_aux)) * 1000.0)
    npt_used, naux = map(int, X_aux_p.shape)
    if (solve_s in fit_metric_qr_methods or solve_s in fit_metric_gram_methods) and npt_used < naux:
        raise ValueError(
            f"THC grid has npt={int(npt_used)} points but aux basis has naux={int(naux)} functions. "
            "Need npt >= naux for the fit_metric solve; increase thc_npt and/or grid size."
        )

    # ---- Aux metric + Cholesky ----
    t_metric = time.perf_counter() if prof is not None else None
    V = cueri_df.metric_2c2e_basis(
        aux_basis,
        stream=stream,
        backend=str(metric_backend),
        mode=str(metric_mode),
        threads=int(metric_threads),
    )
    L = cueri_df.cholesky_metric(V)
    _sync_if_profile()
    if prof is not None and t_metric is not None:
        _profile_ms_set(prof, "metric_ms", (time.perf_counter() - float(t_metric)) * 1000.0)

    # ---- Assemble Z ----
    #
    # Default and recommended: build the point-space Coulomb kernel via the
    # auxiliary metric inverse:
    #   Z = X_aux V^{-1} X_aux^T = (X_aux L^{-T}) (X_aux L^{-T})^T
    #
    # Experimental: fit Z such that X_aux Z X_aux^T ≈ V by solving the
    # underdetermined systems X_aux @ Y = L (npt >= naux), then Z = Y Y^T.
    rcond = 1e-12 if solve_rcond is None else float(solve_rcond)
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("solve_rcond must be a finite, positive float")

    if solve_s in inv_metric_methods:
        t_solve = time.perf_counter() if prof is not None else None
        try:
            import cupyx.scipy.linalg as cpx_linalg  # noqa: PLC0415

            Xw_T = cpx_linalg.solve_triangular(L, X_aux_p.T, lower=True)
        except Exception:
            Xw_T = cp.linalg.solve(L, X_aux_p.T)

        # Y = X_aux V^{-1/2} = X_aux L^{-T}  (npt,naux)
        Y = cp.ascontiguousarray(Xw_T.T)
        Z = cp.ascontiguousarray(Y @ Y.T)
        Z = 0.5 * (Z + Z.T)
        del Xw_T
        _sync_if_profile()
        if prof is not None and t_solve is not None:
            _profile_ms_set(prof, "solve_ms", (time.perf_counter() - float(t_solve)) * 1000.0)
    elif solve_s in fit_metric_qr_methods:
        # Fit V = X_aux Z X_aux^T by solving X_aux @ Y = L, then Z = Y Y^T.
        # Warning: can produce huge values / unstable SCF unless the grid is
        # sufficiently large and well-conditioned.
        t_solve = time.perf_counter() if prof is not None else None
        Q, R = cp.linalg.qr(X_aux_p, mode="reduced")
        # Solve R.T @ G = L (under-determined fit) with SVD regularization on R.
        # This is significantly more robust than a raw triangular solve when R is
        # ill-conditioned (common for large molecules with aggressive point downselect).
        try:
            U, s, Vh = cp.linalg.svd(R, full_matrices=False)
            smax = float(cp.max(s).item()) if int(s.size) else 0.0
            if smax == 0.0:
                inv_s = cp.zeros_like(s)
            else:
                cutoff = float(rcond) * float(smax)
                inv_s = cp.where(s > cutoff, 1.0 / s, 0.0)
            # (R^T)^+ = U diag(inv_s) V^T = U @ (inv_s[:,None] * Vh)
            RinvT = U @ (inv_s[:, None] * Vh)
            G = RinvT @ L
            Y = cp.ascontiguousarray(Q @ G)
            del U, s, Vh, inv_s, RinvT, G
        except Exception:
            # Fallback: attempt an unregularized triangular solve.
            G = cp.linalg.solve(R.T, L)
            Y = cp.ascontiguousarray(Q @ G)
            del G
        Z = cp.ascontiguousarray(Y @ Y.T)
        Z = 0.5 * (Z + Z.T)
        del Q, R
        _sync_if_profile()
        if prof is not None and t_solve is not None:
            _profile_ms_set(prof, "solve_ms", (time.perf_counter() - float(t_solve)) * 1000.0)
    elif solve_s in fit_metric_gram_methods:
        t_solve = time.perf_counter() if prof is not None else None
        Gm = X_aux_p.T @ X_aux_p  # (naux,naux)
        # Ridge regularization using a cutoff scale based on rcond.
        # (Gm eigenvalues are singular_values(X_aux_p)^2)
        try:
            smax = float(cp.linalg.norm(Gm, ord=2).item())
        except Exception:
            smax = 0.0
        lam = (rcond**2) * max(smax, 1.0)
        if lam != 0.0:
            Gm = Gm + lam * cp.eye(int(Gm.shape[0]), dtype=cp.float64)
        G = cp.linalg.solve(Gm, L)
        Y = cp.ascontiguousarray(X_aux_p @ G)
        Z = cp.ascontiguousarray(Y @ Y.T)
        Z = 0.5 * (Z + Z.T)
        del Gm, G
        _sync_if_profile()
        if prof is not None and t_solve is not None:
            _profile_ms_set(prof, "solve_ms", (time.perf_counter() - float(t_solve)) * 1000.0)
    else:
        raise ValueError(
            "solve_method must be one of: "
            "'fit_metric_qr' (default), 'fit_metric_gram', 'inv_metric'. "
            "Aliases: 'qr' -> 'fit_metric_qr', 'gram' -> 'fit_metric_gram'."
        )

    if solve_s in fit_metric_qr_methods or solve_s in fit_metric_gram_methods:
        # Fail fast on pathological fits (typically caused by ill-conditioned grids).
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

    meta = None
    if prof is not None:
        meta = {"profile_key": "thc_factors"}
        try:
            prof["nao"] = int(X.shape[1])
            prof["naux"] = int(naux)
            prof["npt"] = int(X.shape[0])
        except Exception:  # pragma: no cover
            pass
        if t_total is not None:
            _sync_if_profile()
            _profile_ms_set(prof, "total_ms", (time.perf_counter() - float(t_total)) * 1000.0)
    return THCFactors(
        X=X,
        Y=Y,
        Z=Z,
        points=pts,
        weights=w,
        L_metric=L,
        meta=meta,
    )


__all__ = ["THCFactors", "build_thc_factors"]
