from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from .types import GridBatch, GridRequest


def _have_cuda_grid_stack() -> bool:
    """Return True iff the CUDA Becke-partitioning stack is runnable.

    Importing `asuka.density.grids_device` can succeed even when CUDA is not
    usable (it self-disables). For auto-backend decisions we must verify the
    runtime/device availability.
    """

    try:
        from . import grids_device  # noqa: PLC0415

        grids_device._require_cuda_grid_stack()
    except Exception:
        return False
    return True


def _have_cuda_becke() -> bool:
    try:
        from . import grids_device  # noqa: F401, PLC0415
    except Exception:
        return False
    return _have_cuda_grid_stack()


def _have_cuda_rdvr() -> bool:
    if not _have_cuda_grid_stack():
        return False
    try:
        from . import dvr_grids_device  # noqa: F401, PLC0415
    except Exception:
        return False
    return True


def _resolve_backend(req: GridRequest) -> str:
    backend = str(req.backend).strip().lower()
    if backend in {"cpu", "cuda"}:
        return backend

    # auto
    kind = str(req.kind).strip().lower()
    if kind == "becke" and _have_cuda_becke():
        return "cuda"
    if kind == "rdvr" and _have_cuda_rdvr():
        return "cuda"

    # F-DVR currently computes on CPU then may upload.
    return "cpu"


def _xp_concat(arrays: list[Any], *, axis: int = 0):
    if len(arrays) == 0:
        return None
    first = arrays[0]
    mod = type(first).__module__
    if mod.startswith("cupy"):
        import cupy as cp  # noqa: PLC0415

        return cp.ascontiguousarray(cp.concatenate(arrays, axis=axis))
    return np.ascontiguousarray(np.concatenate([np.asarray(a) for a in arrays], axis=axis))


def iter_grid(
    mol_or_coords: Any,
    *,
    request: GridRequest,
    dvr_basis: Any | None = None,
) -> Iterator[GridBatch]:
    """Unified grid iterator across Becke / R-DVR / F-DVR / cube.

    All branches yield `GridBatch`.
    """

    kind = str(request.kind).strip().lower()
    backend = _resolve_backend(request)

    angular_prune = request.angular_prune
    if angular_prune is None:
        angular_prune = bool(kind == "rdvr")

    if kind == "becke":
        if backend == "cuda":
            from .grids_device import iter_becke_grid_device  # noqa: PLC0415

            yield from iter_becke_grid_device(
                mol_or_coords,
                radial_n=int(request.radial_n),
                angular_n=int(request.angular_n),
                angular_kind=str(request.angular_kind),
                rmax=float(request.rmax),
                becke_n=int(request.becke_n),
                block_size=int(request.block_size),
                prune_tol=float(request.prune_tol),
                threads=int(request.threads),
                stream=request.stream,
                angular_prune=bool(angular_prune),
                atom_Z=request.atom_Z,
                return_batch=True,
                radial_scheme="leggauss",  # THC factor fitting uses GL; override function default
            )
            return

        from .grids import iter_becke_grid  # noqa: PLC0415

        yield from iter_becke_grid(
            mol_or_coords,
            radial_n=int(request.radial_n),
            angular_n=int(request.angular_n),
            angular_kind=str(request.angular_kind),
            rmax=float(request.rmax),
            becke_n=int(request.becke_n),
            block_size=int(request.block_size),
            prune_tol=float(request.prune_tol),
            angular_prune=bool(angular_prune),
            atom_Z=request.atom_Z,
            return_batch=True,
            radial_scheme="leggauss",  # THC factor fitting uses GL; override function default
        )
        return

    if kind == "rdvr":
        if dvr_basis is None:
            raise ValueError("kind='rdvr' requires dvr_basis")

        if backend == "cuda":
            from .dvr_grids_device import iter_rdvr_grid_device  # noqa: PLC0415

            yield from iter_rdvr_grid_device(
                mol_or_coords,
                dvr_basis,
                angular_n=int(request.angular_n),
                angular_kind=str(request.angular_kind),
                radial_rmax=float(request.radial_rmax if request.radial_rmax is not None else request.rmax),
                becke_n=int(request.becke_n),
                block_size=int(request.block_size),
                prune_tol=float(request.prune_tol),
                ortho_cutoff=float(request.ortho_cutoff),
                angular_prune=bool(angular_prune),
                atom_Z=request.atom_Z,
                threads=int(request.threads),
                stream=request.stream,
                return_batch=True,
            )
            return

        from .dvr_grids import iter_rdvr_grid  # noqa: PLC0415

        yield from iter_rdvr_grid(
            mol_or_coords,
            dvr_basis,
            angular_n=int(request.angular_n),
            angular_kind=str(request.angular_kind),
            radial_rmax=float(request.radial_rmax if request.radial_rmax is not None else request.rmax),
            becke_n=int(request.becke_n),
            block_size=int(request.block_size),
            prune_tol=float(request.prune_tol),
            ortho_cutoff=float(request.ortho_cutoff),
            angular_prune=bool(angular_prune),
            atom_Z=request.atom_Z,
            return_batch=True,
        )
        return

    if kind == "fdvr":
        if dvr_basis is None:
            raise ValueError("kind='fdvr' requires dvr_basis")

        if backend == "cuda":
            from .dvr_grids_device import iter_fdvr_grid_device  # noqa: PLC0415

            yield from iter_fdvr_grid_device(
                dvr_basis,
                block_size=int(request.block_size),
                ortho_cutoff=float(request.ortho_cutoff),
                max_sweeps=int(request.max_sweeps),
                tol=float(request.tol),
                prune_tol=float(request.prune_tol),
                validate=bool(request.validate),
                overlap_max_abs_tol=float(request.overlap_max_abs_tol),
                return_batch=True,
            )
            return

        from .dvr_grids import iter_fdvr_grid  # noqa: PLC0415

        yield from iter_fdvr_grid(
            dvr_basis,
            block_size=int(request.block_size),
            ortho_cutoff=float(request.ortho_cutoff),
            max_sweeps=int(request.max_sweeps),
            tol=float(request.tol),
            prune_tol=float(request.prune_tol),
            validate=bool(request.validate),
            overlap_max_abs_tol=float(request.overlap_max_abs_tol),
            return_batch=True,
        )
        return

    if kind == "cube":
        from .grids import iter_cube_grid  # noqa: PLC0415

        for pts, w in iter_cube_grid(
            mol_or_coords,
            spacing=float(request.spacing),
            padding=float(request.padding),
            block_size=int(request.block_size),
        ):
            yield GridBatch(points=pts, weights=w, meta={"grid_kind": "cube"})
        return

    raise ValueError("grid kind must be one of: 'becke', 'rdvr', 'fdvr', 'cube'")


def collect_grid(
    mol_or_coords: Any,
    *,
    request: GridRequest,
    dvr_basis: Any | None = None,
) -> GridBatch:
    """Materialize all batches into one `GridBatch`."""

    backend = _resolve_backend(request)
    batches = list(iter_grid(mol_or_coords, request=request, dvr_basis=dvr_basis))
    if len(batches) == 0:
        if backend == "cuda":
            try:
                import cupy as cp  # noqa: PLC0415
            except Exception:
                return GridBatch(
                    points=np.zeros((0, 3), dtype=np.float64),
                    weights=np.zeros((0,), dtype=np.float64),
                    meta={"grid_kind": str(request.kind), "backend": backend, "n_batches": 0},
                )
            return GridBatch(
                points=cp.zeros((0, 3), dtype=cp.float64),
                weights=cp.zeros((0,), dtype=cp.float64),
                meta={"grid_kind": str(request.kind), "backend": backend, "n_batches": 0},
            )
        return GridBatch(
            points=np.zeros((0, 3), dtype=np.float64),
            weights=np.zeros((0,), dtype=np.float64),
            meta={"grid_kind": str(request.kind), "backend": backend, "n_batches": 0},
        )

    pts = _xp_concat([b.points for b in batches], axis=0)
    w = _xp_concat([b.weights for b in batches], axis=0)

    def _concat_optional(name: str):
        vals = [getattr(b, name) for b in batches]
        has_any = any(v is not None for v in vals)
        if not has_any:
            return None
        if not all(v is not None for v in vals):
            raise ValueError(f"inconsistent {name}: some batches have it and others do not")
        return _xp_concat([v for v in vals if v is not None], axis=0)

    point_atom = _concat_optional("point_atom")
    point_radial_index = _concat_optional("point_radial_index")
    point_angular_n = _concat_optional("point_angular_n")

    return GridBatch(
        points=pts,
        weights=w,
        point_atom=point_atom,
        point_radial_index=point_radial_index,
        point_angular_n=point_angular_n,
        meta={
            "grid_kind": str(request.kind),
            "backend": backend,
            "n_batches": int(len(batches)),
        },
    )

