from __future__ import annotations

"""CUDA-backed R-DVR grid construction (CuPy arrays).

This module mirrors :mod:`asuka.density.dvr_grids` but uses the existing CUDA
Becke partitioning kernels in :mod:`asuka._orbitals_cuda_ext` to compute the
molecular partition weights on the GPU.

Design goals
------------
- Keep the expensive Becke partitioning off the CPU.
- Preserve the R-DVR radial node/weight construction from Parrish et al.
  (J. Chem. Phys. 138, 194107 (2013)) by reusing the CPU helper that
  diagonalizes the finite-basis radial coordinate representation.
- Support the same angular pruning heuristic (Bragg-Slater envelope) as the CPU
  implementation.

Notes
-----
- Coordinates are in Bohr.
- All outputs are float64 CuPy arrays.
"""

from typing import Any, Iterator

import numpy as np

from .grids import _coords_bohr, angular_grid
from .grids_device import _get_workspace, _require_cuda_grid_stack, _stream_ptr
from .dvr_grids import (
    _BRAGG_SLATER_RADII_BOHR,
    _atomic_numbers_or_none,
    _lebedev_lmax_from_npts,
    _lebedev_npts_for_lreq,
    _map_shells_to_atoms,
    _pruned_lreq_parrish_2013,
    _rdvr_radial_nodes_weights,
)
from .types import GridBatch


def iter_rdvr_grid_device(
    mol_or_coords: Any,
    dvr_basis: Any,
    *,
    angular_n: int = 302,
    angular_kind: str = "auto",
    radial_rmax: float = 20.0,
    becke_n: int = 3,
    block_size: int = 20000,
    prune_tol: float = 1e-16,
    ortho_cutoff: float = 1e-10,
    angular_prune: bool = True,
    atom_Z: Any | None = None,
    threads: int = 256,
    stream: Any = None,
    return_batch: bool = False,
) -> Iterator[Any]:
    """Yield device ``(points, weights)`` blocks for an R-DVR atom-centered grid.

    This is a CUDA-backed drop-in replacement for
    :func:`asuka.density.dvr_grids.iter_rdvr_grid` with identical semantics, except
    the returned arrays are CuPy arrays on the current CUDA device.

    If ``angular_prune=True`` (default), the Lebedev order is selected *per
    radial node* using the Bragg-Slater envelope from Parrish et al. (2013).

    Parameters
    ----------
    block_size
        Target number of points per yielded block. Use a very large value (e.g.
        ``10**9``) to obtain one block per atom (useful for per-atom downselect).
    """

    cp, ext = _require_cuda_grid_stack()

    R_host = np.asarray(_coords_bohr(mol_or_coords), dtype=np.float64).reshape((-1, 3))
    natm = int(R_host.shape[0])
    if natm <= 0:
        raise ValueError("no atoms")

    angular_n_max = int(angular_n)
    if angular_n_max <= 0:
        raise ValueError("angular_n must be > 0")
    block_size = max(1, int(block_size))
    prune_tol = float(prune_tol)
    becke_n = int(becke_n)
    if becke_n < 0:
        raise ValueError("becke_n must be >= 0")
    threads = int(threads)
    if threads <= 0:
        raise ValueError("threads must be > 0")
    return_batch = bool(return_batch)

    angular_prune = bool(angular_prune)
    angular_kind_s = str(angular_kind).strip().lower()
    if angular_prune and angular_kind_s in {"fibonacci", "fib", "fibo"}:
        raise ValueError("angular_prune=True requires Lebedev grids (use angular_kind='auto' or 'lebedev')")

    # Angular cache: host (NumPy) and device (CuPy).
    ang_host: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    ang_dev: dict[int, tuple[Any, Any]] = {}

    # Pruning metadata (host).
    lmax_max: int | None = None
    rho_bs: np.ndarray | None = None
    if angular_prune:
        lmax_max = _lebedev_lmax_from_npts(int(angular_n_max))
        if lmax_max is None:
            raise ValueError(
                f"angular_prune=True requires angular_n to be a supported Lebedev point count. Got angular_n={int(angular_n_max)}."
            )
        if atom_Z is None:
            Z = _atomic_numbers_or_none(mol_or_coords)
        else:
            Z = np.asarray(atom_Z, dtype=np.int32).ravel()
        if Z is None or int(np.asarray(Z).size) != int(natm):
            raise ValueError(
                "angular_prune=True requires atomic numbers. Pass a Molecule with atoms_bohr or provide atom_Z (len=natm)."
            )
        Z = np.asarray(Z, dtype=np.int32).ravel()
        if int(np.min(Z)) < 1 or int(np.max(Z)) >= int(_BRAGG_SLATER_RADII_BOHR.size):
            raise ValueError("unsupported atomic number for Bragg-Slater pruning")
        rho_bs = np.ascontiguousarray(_BRAGG_SLATER_RADII_BOHR[Z])

    # Shell assignment by atom (host).
    _sh2a, atom_to_shells = _map_shells_to_atoms(np.asarray(dvr_basis.shell_cxyz), R_host)

    # Atom coordinates and inter-atomic distances on device.
    R = cp.ascontiguousarray(cp.asarray(R_host, dtype=cp.float64))
    dAB = R[:, None, :] - R[None, :, :]
    RAB = cp.ascontiguousarray(cp.linalg.norm(dAB, axis=2))

    stream_ptr = _stream_ptr(stream)

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
        dirs_d = cp.ascontiguousarray(cp.asarray(dirs_h, dtype=cp.float64))
        w_d = cp.ascontiguousarray(cp.asarray(w_h, dtype=cp.float64))
        ang_dev[nang_i] = (dirs_d, w_d)
        return dirs_d, w_d

    for ia in range(natm):
        shells = atom_to_shells[int(ia)]
        rho, wr = _rdvr_radial_nodes_weights(
            dvr_basis,
            shells,
            radial_rmax=float(radial_rmax),
            ortho_cutoff=float(ortho_cutoff),
        )
        rho = np.asarray(rho, dtype=np.float64).ravel()
        wr = np.asarray(wr, dtype=np.float64).ravel()
        if int(rho.size) == 0:
            continue

        # Determine Lebedev order per radial node (host).
        nrad = int(rho.size)
        nang_nodes = np.empty((nrad,), dtype=np.int32)
        if angular_prune:
            assert lmax_max is not None
            assert rho_bs is not None
            rho_bs_i = float(rho_bs[int(ia)])
            for i in range(nrad):
                rnode = float(rho[i])
                lreq = _pruned_lreq_parrish_2013(rnode, rho_bs=rho_bs_i, lmax_max=int(lmax_max))
                nang_nodes[i] = int(_lebedev_npts_for_lreq(int(lreq), lmax_max=int(lmax_max)))
        else:
            nang_nodes.fill(int(angular_n_max))

        center = cp.ascontiguousarray(R[int(ia)].reshape((3,)))

        # Buffer blocks per atom to match CPU semantics (flush when n_buf >= block_size).
        pts_buf: list[Any] = []
        w_buf: list[Any] = []
        rid_buf: list[Any] = []
        nang_buf: list[Any] = []
        n_buf = 0

        # Process consecutive segments with identical nang to preserve radial-major ordering.
        i0 = 0
        while i0 < nrad:
            nang = int(nang_nodes[i0])
            i1 = i0 + 1
            while i1 < nrad and int(nang_nodes[i1]) == nang:
                i1 += 1

            i_start = int(i0)
            i_end = int(i1)
            rho_seg = rho[i_start:i_end]
            wr_seg = wr[i_start:i_end]
            i0 = i_end

            dirs_d, wang_d = _get_ang(int(nang))

            radial_r_d = cp.ascontiguousarray(cp.asarray(rho_seg, dtype=cp.float64))
            radial_wr_d = cp.ascontiguousarray(cp.asarray(wr_seg, dtype=cp.float64))

            nloc = int(radial_r_d.shape[0]) * int(nang)
            if nloc <= 0:
                continue

            ws = _get_workspace(max_nloc=max(1, nloc), max_natm=max(1, natm))

            pts_out = cp.empty((nloc, 3), dtype=cp.float64)
            w_out = cp.empty((nloc,), dtype=cp.float64)

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

            mask = None
            if prune_tol > 0.0:
                mask = w_out > float(prune_tol)
                pts_out = pts_out[mask]
                w_out = w_out[mask]

            if int(w_out.size):
                pts_out = cp.ascontiguousarray(pts_out)
                w_out = cp.ascontiguousarray(w_out.ravel())
                rid_out = None
                nang_out = None
                if return_batch:
                    rid_h = np.repeat(np.arange(int(i_start), int(i_end), dtype=np.int32), int(nang))
                    rid_out = cp.ascontiguousarray(cp.asarray(rid_h, dtype=cp.int32))
                    nang_out = cp.full((int(nloc),), int(nang), dtype=cp.int32)
                    if mask is not None:
                        rid_out = rid_out[mask]
                        nang_out = nang_out[mask]
                    rid_out = cp.ascontiguousarray(rid_out.ravel())
                    nang_out = cp.ascontiguousarray(nang_out.ravel())
                pts_buf.append(pts_out)
                w_buf.append(w_out)
                if return_batch:
                    assert rid_out is not None
                    assert nang_out is not None
                    rid_buf.append(rid_out)
                    nang_buf.append(nang_out)
                n_buf += int(w_out.size)

                if n_buf >= int(block_size):
                    if len(pts_buf) == 1:
                        pts_blk = pts_buf[0]
                        w_blk = w_buf[0]
                        rid_blk = rid_buf[0] if return_batch else None
                        nang_blk = nang_buf[0] if return_batch else None
                    else:
                        pts_blk = cp.ascontiguousarray(cp.concatenate(pts_buf, axis=0))
                        w_blk = cp.ascontiguousarray(cp.concatenate(w_buf, axis=0).ravel())
                        rid_blk = cp.ascontiguousarray(cp.concatenate(rid_buf, axis=0).ravel()) if return_batch else None
                        nang_blk = cp.ascontiguousarray(cp.concatenate(nang_buf, axis=0).ravel()) if return_batch else None
                    if return_batch:
                        assert rid_blk is not None
                        assert nang_blk is not None
                        yield GridBatch(
                            points=pts_blk,
                            weights=w_blk,
                            point_atom=cp.full((int(w_blk.size),), int(ia), dtype=cp.int32),
                            point_radial_index=rid_blk,
                            point_angular_n=nang_blk,
                            meta={"grid_kind": "rdvr", "backend": "cuda", "atom": int(ia), "angular_prune": bool(angular_prune)},
                        )
                    else:
                        yield pts_blk, w_blk
                    pts_buf.clear()
                    w_buf.clear()
                    rid_buf.clear()
                    nang_buf.clear()
                    n_buf = 0

        if n_buf:
            if len(pts_buf) == 1:
                pts_blk = pts_buf[0]
                w_blk = w_buf[0]
                rid_blk = rid_buf[0] if return_batch else None
                nang_blk = nang_buf[0] if return_batch else None
            else:
                pts_blk = cp.ascontiguousarray(cp.concatenate(pts_buf, axis=0))
                w_blk = cp.ascontiguousarray(cp.concatenate(w_buf, axis=0).ravel())
                rid_blk = cp.ascontiguousarray(cp.concatenate(rid_buf, axis=0).ravel()) if return_batch else None
                nang_blk = cp.ascontiguousarray(cp.concatenate(nang_buf, axis=0).ravel()) if return_batch else None
            if return_batch:
                assert rid_blk is not None
                assert nang_blk is not None
                yield GridBatch(
                    points=pts_blk,
                    weights=w_blk,
                    point_atom=cp.full((int(w_blk.size),), int(ia), dtype=cp.int32),
                    point_radial_index=rid_blk,
                    point_angular_n=nang_blk,
                    meta={"grid_kind": "rdvr", "backend": "cuda", "atom": int(ia), "angular_prune": bool(angular_prune)},
                )
            else:
                yield pts_blk, w_blk


def iter_fdvr_grid_device(
    dvr_basis: Any,
    *,
    block_size: int = 20_000,
    ortho_cutoff: float = 1e-10,
    max_sweeps: int = 50,
    tol: float = 1e-12,
    prune_tol: float = 1e-16,
    validate: bool = True,
    overlap_max_abs_tol: float = 1e-3,
    return_batch: bool = False,
) -> Iterator[Any]:
    """Yield device ``(points, weights)`` blocks for an F-DVR grid.

    This is a convenience wrapper: build on CPU, upload to GPU, then slice.
    """

    try:
        import cupy as cp  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("iter_fdvr_grid_device requires CuPy") from e

    from .dvr_grids import make_fdvr_grid_device  # noqa: PLC0415

    pts, w = make_fdvr_grid_device(
        dvr_basis,
        ortho_cutoff=float(ortho_cutoff),
        max_sweeps=int(max_sweeps),
        tol=float(tol),
        prune_tol=float(prune_tol),
        validate=bool(validate),
        overlap_max_abs_tol=float(overlap_max_abs_tol),
    )

    block_size = max(1, int(block_size))
    return_batch = bool(return_batch)
    for p0 in range(0, int(w.size), int(block_size)):
        p1 = min(int(w.size), p0 + int(block_size))
        pts_blk = cp.ascontiguousarray(pts[p0:p1])
        w_blk = cp.ascontiguousarray(w[p0:p1].ravel())
        if return_batch:
            yield GridBatch(points=pts_blk, weights=w_blk, meta={"grid_kind": "fdvr", "backend": "cuda"})
        else:
            yield pts_blk, w_blk


def make_rdvr_grid_device(*args: Any, **kwargs: Any):
    """Materialize a full R-DVR grid on the GPU as CuPy arrays."""

    cp, _ext = _require_cuda_grid_stack()
    pts_list: list[Any] = []
    w_list: list[Any] = []
    for pts, w in iter_rdvr_grid_device(*args, block_size=10**9, **kwargs):
        pts_list.append(pts)
        w_list.append(w)
    if len(pts_list) == 0:
        return cp.zeros((0, 3), dtype=cp.float64), cp.zeros((0,), dtype=cp.float64)
    return cp.concatenate(pts_list, axis=0), cp.concatenate(w_list, axis=0).ravel()


__all__ = ["iter_rdvr_grid_device", "make_rdvr_grid_device", "iter_fdvr_grid_device"]
