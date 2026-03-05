from __future__ import annotations

"""CUDA Becke grid construction (CuPy arrays).

This module mirrors :mod:`asuka.density.grids` but keeps points/weights on the
device so downstream GPU kernels can avoid host round-trips.
"""

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from .grids import _coords_bohr, angular_grid, radial_grid_leggauss

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
) -> Iterator[tuple["cp.ndarray", "cp.ndarray"]]:
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

    radial_r, radial_wr = radial_grid_leggauss(radial_n, rmax=float(rmax))
    angular_dirs, angular_w = angular_grid(angular_n, kind=str(angular_kind))

    R = cp_mod.ascontiguousarray(cp_mod.asarray(np.asarray(R_host, dtype=np.float64), dtype=cp_mod.float64))
    radial_r_d = cp_mod.ascontiguousarray(cp_mod.asarray(np.asarray(radial_r, dtype=np.float64), dtype=cp_mod.float64))
    radial_wr_d = cp_mod.ascontiguousarray(
        cp_mod.asarray(np.asarray(radial_wr, dtype=np.float64), dtype=cp_mod.float64)
    )
    angular_dirs_d = cp_mod.ascontiguousarray(
        cp_mod.asarray(np.asarray(angular_dirs, dtype=np.float64), dtype=cp_mod.float64)
    )
    angular_w_d = cp_mod.ascontiguousarray(
        cp_mod.asarray(np.asarray(angular_w, dtype=np.float64), dtype=cp_mod.float64)
    )

    dAB = R[:, None, :] - R[None, :, :]
    RAB = cp_mod.ascontiguousarray(cp_mod.linalg.norm(dAB, axis=2))

    nloc = int(radial_n * angular_n)
    ws = _get_workspace(max_nloc=max(1, nloc), max_natm=max(1, natm))
    stream_ptr = _stream_ptr(stream)

    pts_tmp = cp_mod.empty((nloc, 3), dtype=cp_mod.float64)
    w_tmp = cp_mod.empty((nloc,), dtype=cp_mod.float64)

    for ia in range(natm):
        center = cp_mod.ascontiguousarray(R[ia].reshape((3,)))
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

        if prune_tol > 0.0:
            mask = w_tmp > prune_tol
            pts_atom = pts_tmp[mask]
            w_atom = w_tmp[mask]
        else:
            pts_atom = pts_tmp
            w_atom = w_tmp

        n_this = int(w_atom.shape[0])
        for p0 in range(0, n_this, block_size):
            p1 = min(n_this, p0 + block_size)
            # Return independent blocks to avoid aliasing reusable workspace buffers.
            yield cp_mod.ascontiguousarray(pts_atom[p0:p1].copy()), cp_mod.ascontiguousarray(w_atom[p0:p1].copy())


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
        threads=int(threads),
        stream=stream,
    ):
        pts_list.append(pts)
        w_list.append(w)
    if len(pts_list) == 0:
        return cp_mod.zeros((0, 3), dtype=cp_mod.float64), cp_mod.zeros((0,), dtype=cp_mod.float64)
    return cp_mod.concatenate(pts_list, axis=0), cp_mod.concatenate(w_list, axis=0)


__all__ = ["DeviceGridSpec", "iter_becke_grid_device", "make_becke_grid_device"]
