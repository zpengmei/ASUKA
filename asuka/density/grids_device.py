from __future__ import annotations

"""CUDA Becke grid construction (CuPy arrays).

This module mirrors :mod:`asuka.density.grids` but keeps points/weights on the
device so downstream GPU kernels can avoid host round-trips.
"""

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from .grids import (
    _coords_bohr,
    _plan_becke_angular_orders,
    _radial_grid_for_atom,
    angular_grid,
    radial_grid_leggauss,
)
from .types import GridBatch

try:  # optional CUDA stack
    import cupy as cp  # type: ignore

    from asuka import _orbitals_cuda_ext as _ext  # type: ignore

    _CUDA_GRID_OK = True
except Exception:  # pragma: no cover
    cp = None  # type: ignore
    _ext = None  # type: ignore
    _CUDA_GRID_OK = False


@dataclass(frozen=True)
class DeviceGridSpec:
    """Specification for CUDA Becke grid construction."""

    radial_n: int = 50
    angular_n: int = 302
    angular_kind: str = "auto"
    rmax: float = 20.0
    becke_n: int = 3
    block_size: int = 20000
    prune_tol: float = 1e-16
    threads: int = 256
    # Radial grid scheme: 'leggauss' (fixed [0,rmax] GL) or 'treutler' (Treutler-Ahlrichs M4,
    # per-atom Bragg-Slater scaling — recommended for DFT with meta-GGA functionals).
    radial_scheme: str = "treutler"


_GRID_WS_CACHE: dict[int, tuple[int, int, Any]] = {}


def _require_cuda_grid_stack():
    if not _CUDA_GRID_OK:  # pragma: no cover
        raise RuntimeError("CUDA grid backend unavailable (requires cupy and asuka._orbitals_cuda_ext)")
    assert cp is not None
    assert _ext is not None
    if int(cp.cuda.runtime.getDeviceCount()) <= 0:  # pragma: no cover
        raise RuntimeError("CUDA grid backend requested but no CUDA devices are visible")
    return cp, _ext


def _stream_ptr(stream: Any) -> int:
    if stream is None:
        assert cp is not None
        return int(cp.cuda.get_current_stream().ptr)
    if hasattr(stream, "ptr"):
        return int(stream.ptr)
    return int(stream)


def _get_workspace(*, max_nloc: int, max_natm: int):
    cp_mod, ext = _require_cuda_grid_stack()
    dev = int(cp_mod.cuda.runtime.getDevice())
    entry = _GRID_WS_CACHE.get(dev)
    if entry is not None:
        old_nloc, old_natm, ws = entry
        if old_nloc >= int(max_nloc) and old_natm >= int(max_natm):
            return ws
        try:
            ws.release()
        except Exception:  # pragma: no cover
            pass
    ws = ext.BeckeGridWorkspace(int(max_nloc), int(max_natm))
    _GRID_WS_CACHE[dev] = (int(max_nloc), int(max_natm), ws)
    return ws


def iter_becke_grid_device(
    mol_or_coords: Any,
    *,
    radial_n: int = 50,
    angular_n: int = 302,
    angular_kind: str = "auto",
    rmax: float = 20.0,
    becke_n: int = 3,
    block_size: int = 20000,
    prune_tol: float = 1e-16,
    threads: int = 256,
    stream: Any = None,
    angular_prune: bool = False,
    atom_Z: Any | None = None,
    return_batch: bool = False,
    radial_scheme: str = "leggauss",
) -> Iterator[Any]:
    """Yield device `(points, weights)` blocks for an atom-centered Becke grid.

    Notes
    -----
    - Point ordering matches CPU implementation: atom-major, radial-major,
      angular-major, prune-stable.
    - Arrays are float64 CuPy arrays in Bohr units.
    """

    cp_mod, ext = _require_cuda_grid_stack()

    R_host = _coords_bohr(mol_or_coords)
    natm = int(R_host.shape[0])
    if natm <= 0:
        raise ValueError("no atoms")

    radial_n = int(radial_n)
    angular_n = int(angular_n)
    becke_n = int(becke_n)
    block_size = max(1, int(block_size))
    threads = int(threads)
    prune_tol = float(prune_tol)
    if radial_n <= 0 or angular_n <= 0:
        raise ValueError("radial_n and angular_n must be > 0")
    if becke_n < 0:
        raise ValueError("becke_n must be >= 0")
    if threads <= 0:
        raise ValueError("threads must be > 0")

    R = cp_mod.ascontiguousarray(cp_mod.asarray(np.asarray(R_host, dtype=np.float64), dtype=cp_mod.float64))

    dAB = R[:, None, :] - R[None, :, :]
    RAB = cp_mod.ascontiguousarray(cp_mod.linalg.norm(dAB, axis=2))

    stream_ptr = _stream_ptr(stream)

    angular_prune = bool(angular_prune)
    return_batch = bool(return_batch)
    radial_scheme = str(radial_scheme).strip().lower()

    # Resolve atomic numbers when needed.
    needs_Z = angular_prune or radial_scheme not in ("leggauss", "gl", "gauss-legendre")
    Z_all: np.ndarray | None = None
    if needs_Z:
        if atom_Z is None:
            from .dvr_grids import _atomic_numbers_or_none  # noqa: PLC0415

            Z_all = _atomic_numbers_or_none(mol_or_coords)
        else:
            Z_all = np.asarray(atom_Z, dtype=np.int32).ravel()
        if Z_all is None:
            what = "angular_prune=True" if angular_prune else f"radial_scheme={radial_scheme!r}"
            raise ValueError(
                f"{what} requires atomic numbers. Pass a Molecule with atoms_bohr or provide atom_Z (len=natm)."
            )
        Z_all = np.asarray(Z_all, dtype=np.int32).ravel()
        if int(Z_all.size) == 1 and natm > 1:
            Z_all = np.full((natm,), int(Z_all[0]), dtype=np.int32)
        if int(Z_all.size) != int(natm):
            raise ValueError("atom_Z must have length natm when provided")

    # For leggauss, precompute a shared host-side radial grid.
    _radial_r_shared: np.ndarray | None = None
    _radial_wr_shared: np.ndarray | None = None
    if radial_scheme in ("leggauss", "gl", "gauss-legendre"):
        _radial_r_shared, _radial_wr_shared = radial_grid_leggauss(radial_n, rmax=float(rmax))

    # Fast path: fixed angular order everywhere (no per-radial pruning).
    if not angular_prune:
        angular_dirs, angular_w = angular_grid(angular_n, kind=str(angular_kind))
        angular_dirs_d = cp_mod.ascontiguousarray(
            cp_mod.asarray(np.asarray(angular_dirs, dtype=np.float64), dtype=cp_mod.float64)
        )
        angular_w_d = cp_mod.ascontiguousarray(cp_mod.asarray(np.asarray(angular_w, dtype=np.float64), dtype=cp_mod.float64))

        nloc = int(radial_n * angular_n)
        ws = _get_workspace(max_nloc=max(1, nloc), max_natm=max(1, natm))

        pts_tmp = cp_mod.empty((nloc, 3), dtype=cp_mod.float64)
        w_tmp = cp_mod.empty((nloc,), dtype=cp_mod.float64)

        # For leggauss, upload shared radial arrays once.
        _radial_r_d_shared: Any | None = None
        _radial_wr_d_shared: Any | None = None
        if _radial_r_shared is not None:
            _radial_r_d_shared = cp_mod.ascontiguousarray(cp_mod.asarray(np.asarray(_radial_r_shared, dtype=np.float64), dtype=cp_mod.float64))
            _radial_wr_d_shared = cp_mod.ascontiguousarray(cp_mod.asarray(np.asarray(_radial_wr_shared, dtype=np.float64), dtype=cp_mod.float64))
            rid_base_shared = cp_mod.ascontiguousarray(cp_mod.asarray(
                np.repeat(np.arange(int(radial_n), dtype=np.int32), int(angular_n)), dtype=cp_mod.int32
            )) if return_batch else None
            nang_base_shared = cp_mod.full((int(nloc),), int(angular_n), dtype=cp_mod.int32) if return_batch else None

        for ia in range(natm):
            if _radial_r_d_shared is not None:
                radial_r_d = _radial_r_d_shared
                radial_wr_d = _radial_wr_d_shared
                rid_base = rid_base_shared
                nang_base = nang_base_shared
            else:
                Z_ia = int(Z_all[int(ia)]) if Z_all is not None else None  # type: ignore[index]
                r_ia, wr_ia = _radial_grid_for_atom(radial_n, Z_ia, scheme=radial_scheme, rmax=float(rmax))
                radial_r_d = cp_mod.ascontiguousarray(cp_mod.asarray(np.asarray(r_ia, dtype=np.float64), dtype=cp_mod.float64))
                radial_wr_d = cp_mod.ascontiguousarray(cp_mod.asarray(np.asarray(wr_ia, dtype=np.float64), dtype=cp_mod.float64))
                if return_batch:
                    rid_base = cp_mod.ascontiguousarray(cp_mod.asarray(
                        np.repeat(np.arange(int(radial_n), dtype=np.int32), int(angular_n)), dtype=cp_mod.int32
                    ))
                    nang_base = cp_mod.full((int(nloc),), int(angular_n), dtype=cp_mod.int32)
                else:
                    rid_base = None
                    nang_base = None

            center = cp_mod.ascontiguousarray(R[int(ia)].reshape((3,)))
            ext.eval_becke_atom_block_f64_inplace_device(
                center,
                radial_r_d,
                radial_wr_d,
                angular_dirs_d,
                angular_w_d,
                R,
                RAB,
                int(ia),
                int(becke_n),
                pts_tmp,
                w_tmp,
                ws,
                threads=int(threads),
                stream_ptr=int(stream_ptr),
                sync=False,
            )

            rid_atom = rid_base
            nang_atom = nang_base
            if prune_tol > 0.0:
                mask = w_tmp > prune_tol
                pts_atom = pts_tmp[mask]
                w_atom = w_tmp[mask]
                if rid_base is not None:
                    rid_atom = rid_base[mask]
                if nang_base is not None:
                    nang_atom = nang_base[mask]
            else:
                pts_atom = pts_tmp
                w_atom = w_tmp

            n_this = int(w_atom.shape[0])
            for p0 in range(0, n_this, int(block_size)):
                p1 = min(n_this, p0 + int(block_size))
                pts_blk = cp_mod.ascontiguousarray(pts_atom[p0:p1].copy())
                w_blk = cp_mod.ascontiguousarray(w_atom[p0:p1].copy()).ravel()
                if return_batch:
                    assert rid_atom is not None
                    assert nang_atom is not None
                    rid_blk = cp_mod.ascontiguousarray(rid_atom[p0:p1].copy())
                    nang_blk = cp_mod.ascontiguousarray(nang_atom[p0:p1].copy())
                    yield GridBatch(
                        points=pts_blk,
                        weights=w_blk,
                        point_atom=cp_mod.full((int(w_blk.size),), int(ia), dtype=cp_mod.int32),
                        point_radial_index=rid_blk,
                        point_angular_n=nang_blk,
                        meta={"grid_kind": "becke", "backend": "cuda", "atom": int(ia), "angular_prune": False},
                    )
                else:
                    yield pts_blk, w_blk
        return

    # Pruned path: variable angular order per radial node (Lebedev-only).
    # Z_all already resolved above (since angular_prune implies needs_Z=True).
    assert Z_all is not None  # for type-checkers

    ang_host: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    ang_dev: dict[int, tuple[Any, Any]] = {}

    def _get_ang(nang: int) -> tuple[Any, Any]:
        nang_i = int(nang)
        hit = ang_dev.get(nang_i)
        if hit is not None:
            return hit
        h = ang_host.get(nang_i)
        if h is None:
            dirs, w = angular_grid(int(nang_i), kind=str(angular_kind))
            dirs = np.asarray(dirs, dtype=np.float64).reshape((-1, 3))
            w = np.asarray(w, dtype=np.float64).ravel()
            if int(dirs.shape[0]) != int(w.shape[0]):
                raise RuntimeError("angular grid returned inconsistent shapes")
            h = (dirs, w)
            ang_host[nang_i] = h
        dirs_h, w_h = h
        dirs_d = cp_mod.ascontiguousarray(cp_mod.asarray(dirs_h, dtype=cp_mod.float64))
        w_d = cp_mod.ascontiguousarray(cp_mod.asarray(w_h, dtype=cp_mod.float64))
        ang_dev[nang_i] = (dirs_d, w_d)
        return dirs_d, w_d

    for ia in range(natm):
        Z_ia = int(Z_all[int(ia)])
        # Per-atom radial grid (TA or GL).
        if _radial_r_shared is not None:
            radial_r = _radial_r_shared
            radial_wr = _radial_wr_shared
        else:
            radial_r, radial_wr = _radial_grid_for_atom(radial_n, Z_ia, scheme=radial_scheme, rmax=float(rmax))
        nang_nodes = _plan_becke_angular_orders(
            radial_r,
            angular_n_max=int(angular_n),
            angular_kind=str(angular_kind),
            angular_prune=True,
            atom_Z=Z_ia,
        )

        pts_parts: list[Any] = []
        w_parts: list[Any] = []
        rid_parts: list[Any] = []
        nang_parts: list[Any] = []

        center = cp_mod.ascontiguousarray(R[int(ia)].reshape((3,)))
        i0 = 0
        while i0 < int(radial_r.size):
            nang = int(nang_nodes[i0])
            i1 = i0 + 1
            while i1 < int(radial_r.size) and int(nang_nodes[i1]) == nang:
                i1 += 1

            rho_seg = radial_r[i0:i1]
            wr_seg = radial_wr[i0:i1]
            i_start = int(i0)
            i_end = int(i1)
            i0 = i_end

            radial_r_d = cp_mod.ascontiguousarray(cp_mod.asarray(np.asarray(rho_seg, dtype=np.float64), dtype=cp_mod.float64))
            radial_wr_d = cp_mod.ascontiguousarray(cp_mod.asarray(np.asarray(wr_seg, dtype=np.float64), dtype=cp_mod.float64))
            dirs_d, wang_d = _get_ang(int(nang))

            nloc = int(radial_r_d.shape[0]) * int(nang)
            if nloc <= 0:
                continue
            ws = _get_workspace(max_nloc=max(1, nloc), max_natm=max(1, natm))

            pts_out = cp_mod.empty((nloc, 3), dtype=cp_mod.float64)
            w_out = cp_mod.empty((nloc,), dtype=cp_mod.float64)

            ext.eval_becke_atom_block_f64_inplace_device(
                center,
                radial_r_d,
                radial_wr_d,
                dirs_d,
                wang_d,
                R,
                RAB,
                int(ia),
                int(becke_n),
                pts_out,
                w_out,
                ws,
                threads=int(threads),
                stream_ptr=int(stream_ptr),
                sync=False,
            )

            rid_out = None
            nang_out = None
            if return_batch:
                rid_h = np.repeat(np.arange(int(i_start), int(i_end), dtype=np.int32), int(nang))
                rid_out = cp_mod.ascontiguousarray(cp_mod.asarray(rid_h, dtype=cp_mod.int32))
                nang_out = cp_mod.full((int(nloc),), int(nang), dtype=cp_mod.int32)

            if prune_tol > 0.0:
                mask = w_out > float(prune_tol)
                pts_out = pts_out[mask]
                w_out = w_out[mask]
                if rid_out is not None:
                    rid_out = rid_out[mask]
                if nang_out is not None:
                    nang_out = nang_out[mask]

            if int(w_out.size):
                pts_parts.append(cp_mod.ascontiguousarray(pts_out))
                w_parts.append(cp_mod.ascontiguousarray(w_out).ravel())
                if return_batch:
                    assert rid_out is not None
                    assert nang_out is not None
                    rid_parts.append(cp_mod.ascontiguousarray(rid_out))
                    nang_parts.append(cp_mod.ascontiguousarray(nang_out))

        if len(w_parts) == 0:
            continue

        pts_atom = pts_parts[0] if len(pts_parts) == 1 else cp_mod.ascontiguousarray(cp_mod.concatenate(pts_parts, axis=0))
        w_atom = w_parts[0] if len(w_parts) == 1 else cp_mod.ascontiguousarray(cp_mod.concatenate(w_parts, axis=0)).ravel()
        rid_atom = None
        nang_atom = None
        if return_batch:
            rid_atom = rid_parts[0] if len(rid_parts) == 1 else cp_mod.ascontiguousarray(cp_mod.concatenate(rid_parts, axis=0)).ravel()
            nang_atom = nang_parts[0] if len(nang_parts) == 1 else cp_mod.ascontiguousarray(cp_mod.concatenate(nang_parts, axis=0)).ravel()

        n_this = int(w_atom.shape[0])
        for p0 in range(0, n_this, int(block_size)):
            p1 = min(n_this, p0 + int(block_size))
            pts_blk = cp_mod.ascontiguousarray(pts_atom[p0:p1].copy())
            w_blk = cp_mod.ascontiguousarray(w_atom[p0:p1].copy()).ravel()
            if return_batch:
                assert rid_atom is not None
                assert nang_atom is not None
                rid_blk = cp_mod.ascontiguousarray(rid_atom[p0:p1].copy())
                nang_blk = cp_mod.ascontiguousarray(nang_atom[p0:p1].copy())
                yield GridBatch(
                    points=pts_blk,
                    weights=w_blk,
                    point_atom=cp_mod.full((int(w_blk.size),), int(ia), dtype=cp_mod.int32),
                    point_radial_index=rid_blk,
                    point_angular_n=nang_blk,
                    meta={"grid_kind": "becke", "backend": "cuda", "atom": int(ia), "angular_prune": True},
                )
            else:
                yield pts_blk, w_blk


def make_becke_grid_device(
    mol_or_coords: Any,
    *,
    radial_n: int = 50,
    angular_n: int = 302,
    angular_kind: str = "auto",
    rmax: float = 20.0,
    becke_n: int = 3,
    prune_tol: float = 1e-16,
    threads: int = 256,
    stream: Any = None,
    radial_scheme: str = "leggauss",
    atom_Z: Any | None = None,
) -> tuple["cp.ndarray", "cp.ndarray"]:
    """Materialize a full CUDA Becke grid as CuPy arrays."""

    cp_mod, _ext_mod = _require_cuda_grid_stack()
    pts_list: list[cp.ndarray] = []
    w_list: list[cp.ndarray] = []
    for pts, w in iter_becke_grid_device(
        mol_or_coords,
        radial_n=int(radial_n),
        angular_n=int(angular_n),
        angular_kind=str(angular_kind),
        rmax=float(rmax),
        becke_n=int(becke_n),
        block_size=10**9,
        prune_tol=float(prune_tol),
        radial_scheme=str(radial_scheme),
        atom_Z=atom_Z,
        threads=int(threads),
        stream=stream,
    ):
        pts_list.append(pts)
        w_list.append(w)
    if len(pts_list) == 0:
        return cp_mod.zeros((0, 3), dtype=cp_mod.float64), cp_mod.zeros((0,), dtype=cp_mod.float64)
    return cp_mod.concatenate(pts_list, axis=0), cp_mod.concatenate(w_list, axis=0)


__all__ = ["DeviceGridSpec", "iter_becke_grid_device", "make_becke_grid_device"]
