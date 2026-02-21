from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.caspt2.f3 import CASPT2CIContext
from asuka.cuda import cuda_backend


def _require_f3_cuda_deps() -> None:
    if not cuda_backend.has_cuda_ext():
        raise RuntimeError(
            "cuda_mode='full' requires the GUGA CUDA extension. "
            "Build it with: python -m asuka.build.guga_cuda_ext"
        )
    if not cuda_backend.has_epq_table_device_build():
        raise RuntimeError(
            "cuda_mode='full' requires EPQ-table device-build entrypoints, but they are missing. "
            "Rebuild the GUGA CUDA extension: python -m asuka.build.guga_cuda_ext"
        )


@dataclass
class F3CudaWorkspace:
    """Reusable device-side structures for EPQ-table-based E_pq applications."""

    key: tuple[int, int, int]  # (device_id, norb, ncsf)

    norb: int
    ncsf: int
    nops: int

    drt_dev: Any
    state_dev: Any
    epq_table: Any
    epq_table_t: Any
    overflow: Any

    occ: Any | None = None  # (ncsf, norb) float64


def _get_device_id(cp) -> int:
    try:
        return int(cp.cuda.runtime.getDevice())
    except Exception:
        return int(cp.cuda.Device().id)


def get_f3_cuda_workspace(drt, *, device: int | None = None, threads: int = 256) -> F3CudaWorkspace:
    """Create or reuse a cached F3 workspace attached to `drt` for the current device."""
    _require_f3_cuda_deps()

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for cuda_mode='full'") from e

    if device is not None:
        cp.cuda.Device(int(device)).use()

    device_id = _get_device_id(cp)
    key = (int(device_id), int(drt.norb), int(drt.ncsf))
    cached = getattr(drt, "_caspt2_f3_cuda_workspace", None)
    if cached is not None and getattr(cached, "key", None) == key:
        return cached

    drt_dev = cuda_backend.make_device_drt(drt)
    state_dev = cuda_backend.make_device_state_cache(drt, drt_dev)
    epq_table = cuda_backend.build_epq_action_table_combined_device(
        drt,
        drt_dev,
        state_dev,
        threads=int(threads),
        use_cache=True,
        sync=True,
        check_overflow=True,
    )
    epq_table_t = cuda_backend.build_epq_action_table_transpose_device(
        drt,
        epq_table,
        use_cache=True,
        validate=False,
    )
    overflow = cp.empty((1,), dtype=cp.int32)

    ws = F3CudaWorkspace(
        key=key,
        norb=int(drt.norb),
        ncsf=int(drt.ncsf),
        nops=int(drt.norb) * int(drt.norb),
        drt_dev=drt_dev,
        state_dev=state_dev,
        epq_table=epq_table,
        epq_table_t=epq_table_t,
        overflow=overflow,
        occ=None,
    )
    setattr(drt, "_caspt2_f3_cuda_workspace", ws)
    return ws


def epq_apply_all_cuda(
    drt,
    ws: F3CudaWorkspace,
    *,
    x,
    out: Any | None = None,
    threads: int = 256,
) -> Any:
    """Compute W[:,pq] = E_pq |x> for all pq (including diagonal E_pp) on GPU."""
    _require_f3_cuda_deps()

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for cuda_mode='full'") from e

    ncsf = int(ws.ncsf)
    nops = int(ws.nops)

    x_d = cp.asarray(x, dtype=cp.float64).ravel()
    x_d = cp.ascontiguousarray(x_d)
    if x_d.shape != (ncsf,):
        raise ValueError(f"x must have shape ({ncsf},)")

    if out is None:
        w = cp.zeros((ncsf, nops), dtype=cp.float64)
    else:
        w = cp.asarray(out, dtype=cp.float64)
        if w.shape != (ncsf, nops):
            raise ValueError(f"out must have shape ({ncsf},{nops})")
        if not getattr(w, "flags", None) or not w.flags.c_contiguous:
            w = cp.ascontiguousarray(w)
        w.fill(0.0)

    # Diagonal E_pp contributions: W[j,pp] = x[j] * occ(j,p).
    cuda_backend.build_w_diag_from_steps_inplace_device(
        ws.state_dev,
        j_start=0,
        j_count=ncsf,
        x=x_d,
        w_out=w,
        threads=int(threads),
        sync=False,
        relative_w=False,
    )

    # Off-diagonal contributions from EPQ table.
    w, overflow = cuda_backend.build_w_from_epq_table_inplace_device(
        drt,
        ws.state_dev,
        ws.epq_table,
        x_d,
        w_out=w,
        overflow=ws.overflow,
        threads=int(threads),
        sync=True,
        check_overflow=True,
    )
    ov = int(cp.asnumpy(overflow[0]))
    if ov != 0:
        raise RuntimeError(f"EPQ apply overflow on GPU (overflow={ov}); reduce problem size or rebuild EPQ tables")
    return w


def epq_apply_all_cuda_transpose_range(
    drt,
    ws: F3CudaWorkspace,
    *,
    x,
    out: Any,
    k_start: int,
    k_count: int,
    threads: int = 256,
    stream=None,
    sync: bool = False,
    check_overflow: bool = False,
) -> Any:
    """Compute W_block[:,pq] = E_pq |x> for k in [k_start, k_start+k_count) on GPU.

    Uses destination-major EPQ transpose tables (gather path) to reduce atomic contention.
    `out` must have shape (k_count, nops) and will be overwritten.
    """
    _require_f3_cuda_deps()

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for cuda_mode='full'") from e

    ncsf = int(ws.ncsf)
    nops = int(ws.nops)
    k_start = int(k_start)
    k_count = int(k_count)
    if k_start < 0 or k_start >= ncsf:
        raise ValueError("k_start out of range")
    if k_count <= 0 or k_start + k_count > ncsf:
        raise ValueError("k_count out of range")

    x_d = cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64).ravel())
    if x_d.shape != (ncsf,):
        raise ValueError(f"x must have shape ({ncsf},)")

    w = cp.asarray(out, dtype=cp.float64)
    if w.shape != (k_count, nops):
        raise ValueError(f"out must have shape ({k_count},{nops})")
    if not getattr(w, "flags", None) or not w.flags.c_contiguous:
        w = cp.ascontiguousarray(w)

    if stream is None:
        stream = cp.cuda.get_current_stream()

    # Reset output block (fast memset).
    cp.cuda.runtime.memsetAsync(int(w.data.ptr), 0, int(w.size) * int(w.itemsize), int(stream.ptr))

    # Diagonal E_pp contributions: W[k,pp] = x[k] * occ(k,p).
    cuda_backend.build_w_diag_from_steps_inplace_device(
        ws.state_dev,
        j_start=int(k_start),
        j_count=int(k_count),
        x=x_d,
        w_out=w,
        threads=int(threads),
        stream=stream,
        sync=False,
        relative_w=True,
    )

    # Off-diagonal contributions from destination-major transpose table.
    cuda_backend.build_w_from_epq_transpose_range_inplace_device(
        drt,
        ws.state_dev,
        ws.epq_table_t,
        x_d,
        w_out=w,
        overflow=ws.overflow,
        threads=int(threads),
        stream=stream,
        sync=bool(sync),
        check_overflow=bool(check_overflow),
        k_start=int(k_start),
        k_count=int(k_count),
    )
    return w


class F3ContractionEngineCuda:
    """GPU analogue of `asuka.caspt2.f3.F3ContractionEngine` for CASPT2 cases A/C."""

    def __init__(
        self,
        context: CASPT2CIContext,
        epsa: np.ndarray,
        *,
        device: int | None = None,
        cache_bytes: int = 512 * 1024 * 1024,
        build_threads: int = 256,
        apply_threads: int = 256,
    ) -> None:
        _require_f3_cuda_deps()

        try:
            import cupy as cp
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CuPy is required for cuda_mode='full'") from e

        if device is not None:
            cp.cuda.Device(int(device)).use()

        self._cp = cp
        self.drt = context.drt
        self.norb = int(self.drt.norb)
        self.ncsf = int(self.drt.ncsf)
        self.nops = int(self.norb) * int(self.norb)

        c = np.asarray(context.ci_csf, dtype=np.float64).ravel()
        if c.shape != (self.ncsf,):
            raise ValueError("ci_csf has wrong length for provided DRT")
        self.c = cp.ascontiguousarray(cp.asarray(c, dtype=cp.float64))

        epsa_h = np.asarray(epsa, dtype=np.float64).ravel()
        if epsa_h.shape != (self.norb,):
            raise ValueError(f"epsa shape {epsa_h.shape} incompatible with norb={self.norb}")
        self.epsa = cp.ascontiguousarray(cp.asarray(epsa_h, dtype=cp.float64))

        self._ws = get_f3_cuda_workspace(self.drt, device=device, threads=int(build_threads))
        self._apply_threads = int(apply_threads)
        self._tile_csf = int(os.getenv("ASUKA_CASPT2_F3_TILE_CSF", "8192").strip() or "8192")
        if self._tile_csf <= 0:
            self._tile_csf = 8192
        self._tile_csf = max(1, min(int(self._tile_csf), int(self.ncsf)))
        self._use_tiled = bool(int(os.getenv("ASUKA_CASPT2_F3_USE_TILED", "0").strip() or "0"))
        self._check_overflow = bool(int(os.getenv("ASUKA_CASPT2_F3_CHECK_OVERFLOW", "0").strip() or "0"))
        self._w_tile = None
        self._k_csf = None

        # Occupancy table (cached in workspace).
        if self._ws.occ is None:
            self._ws.occ = cuda_backend.build_occ_block_from_steps_inplace_device(
                self._ws.state_dev,
                j_start=0,
                j_count=self.ncsf,
                occ_out=None,
                threads=256,
                sync=True,
            )
        self.occ = self._ws.occ

        # hdiag[j] = sum_w epsa[w] * occ(j,w)
        self.hdiag = cp.ascontiguousarray(self.occ @ self.epsa)
        self.fc = cp.ascontiguousarray(self.c * self.hdiag)

        # T1[j,pq] = (E_pq |c>)[j], and similarly for |fc>.
        self.t1_csf = epq_apply_all_cuda(
            self.drt,
            self._ws,
            x=self.c,
            threads=int(self._apply_threads),
        )
        self.t1_fc_csf = epq_apply_all_cuda(
            self.drt,
            self._ws,
            x=self.fc,
            threads=int(self._apply_threads),
        )

        # Bra buffer: B[j, pq] = (E_qp |c>)[j] so that B[:,pq]^T @ x = <c|E_pq|x>.
        ids = cp.arange(self.nops, dtype=cp.int32)
        p = ids // int(self.norb)
        q = ids - p * int(self.norb)
        qp_ids = q * int(self.norb) + p
        self.bra_csf = cp.ascontiguousarray(self.t1_csf[:, qp_ids])

        self._yz_cache: dict[int, Any] = {}
        self._yz_cache_f3raw: dict[int, Any] = {}
        self._cache_budget = int(max(0, int(cache_bytes)))
        self._cache_bytes = 0
        self._f3raw_all_precomputed = False
        self._w_tile_mm: dict[int, Any] = {}  # key: out_cols (=nops*nvec)

    def precompute_f3raw_all(self) -> None:
        """Precompute and cache all F3-raw yz matrices for this (DRT, CI, epsa) context."""
        cp = self._cp

        if self._f3raw_all_precomputed:
            return
        if len(self._yz_cache_f3raw) == int(self.nops):
            self._f3raw_all_precomputed = True
            return
        if int(self._cache_budget) <= 0:
            # No caching requested; keep old on-demand behavior.
            self._f3raw_all_precomputed = True
            return

        if not cuda_backend.has_build_w_from_epq_transpose_range_mm_scaled():
            # Fallback: compute on-demand with the existing scalar path.
            self._f3raw_all_precomputed = True
            return

        # y,z batching is bounded by shared memory: nops*nvec*sizeof(double).
        yz_batch = int(os.getenv("ASUKA_CASPT2_F3_YZ_BATCH", "8").strip() or "8")
        yz_batch = max(1, yz_batch)
        max_smem_bytes = 48 * 1024
        max_batch = max(1, int(max_smem_bytes // (int(self.nops) * 8)))
        yz_batch = min(yz_batch, max_batch)

        ncsf = int(self.ncsf)
        norb = int(self.norb)
        nops = int(self.nops)
        tile_csf = int(self._tile_csf)

        # Force caching to stay within budget (typical CAS(12,12) is small enough).
        for yz0 in range(0, nops, yz_batch):
            yz1 = min(nops, int(yz0) + int(yz_batch))
            nvec = int(yz1 - yz0)
            out_cols = int(nops) * int(nvec)

            # x_batch[j,vec] = (E_(yz)|c>)[j] for yz in [yz0,yz1).
            x_batch = self.t1_csf[:, int(yz0) : int(yz1)]

            mat_batch = cp.zeros((nops, out_cols), dtype=cp.float64)

            # Reusable W tile buffer for this out_cols.
            w_tile_full = self._w_tile_mm.get(out_cols)
            if w_tile_full is None or tuple(getattr(w_tile_full, "shape", ())) != (tile_csf, out_cols):
                w_tile_full = cp.empty((tile_csf, out_cols), dtype=cp.float64)
                self._w_tile_mm[out_cols] = w_tile_full

            for k_start in range(0, ncsf, tile_csf):
                k_count = min(tile_csf, ncsf - int(k_start))
                w_tile = w_tile_full[: int(k_count)]

                if self._check_overflow:
                    self._ws.overflow.fill(0)

                cuda_backend.build_w_from_epq_transpose_range_mm_scaled_inplace_device(
                    self.drt,
                    self._ws.state_dev,
                    self._ws.epq_table_t,
                    x_batch,
                    self.hdiag,
                    self.epsa,
                    w_out=w_tile,
                    overflow=self._ws.overflow,
                    threads=int(self._apply_threads),
                    stream=cp.cuda.get_current_stream(),
                    sync=False,
                    check_overflow=False,
                    k_start=int(k_start),
                    k_count=int(k_count),
                )

                bra_blk = self.bra_csf[int(k_start) : int(k_start) + int(k_count), :]
                mat_batch += bra_blk.T @ w_tile

            if self._check_overflow:
                try:
                    cp.cuda.get_current_stream().synchronize()
                except Exception:
                    pass
                ov = int(cp.asnumpy(self._ws.overflow[0]))
                if ov != 0:
                    raise RuntimeError(
                        f"EPQ apply overflow on GPU (overflow={ov}); reduce problem size or rebuild EPQ tables"
                    )

            # Store each yz matrix from the batch.
            mat_batch_3 = mat_batch.reshape(nops, nops, nvec, order="C")
            for lane in range(nvec):
                yz_id = int(yz0) + int(lane)
                if yz_id in self._yz_cache_f3raw:
                    continue
                mat = cp.ascontiguousarray(mat_batch_3[:, :, int(lane)])
                if self._cache_budget > 0:
                    nb = int(getattr(mat, "nbytes", 0))
                    if self._cache_bytes + nb <= self._cache_budget:
                        self._yz_cache_f3raw[int(yz_id)] = mat
                        self._cache_bytes += nb
                # If cache budget is 0, do not retain (fallback to on-demand compute).

        self._f3raw_all_precomputed = True

    def matrix_for_yz(self, y: int, z: int) -> Any:
        """Return M[pq,vx] = <c|E_pq E_vx E_yz|fc> for fixed (y,z)."""
        cp = self._cp
        y = int(y)
        z = int(z)
        if y < 0 or y >= int(self.norb) or z < 0 or z >= int(self.norb):
            raise ValueError("y/z out of range")
        yz = y * int(self.norb) + z
        cached = self._yz_cache.get(yz)
        if cached is not None:
            return cached

        fyz = self.t1_fc_csf[:, yz]
        k_csf = epq_apply_all_cuda(
            self.drt,
            self._ws,
            x=fyz,
            threads=int(self._apply_threads),
        )
        mat = cp.ascontiguousarray(self.bra_csf.T @ k_csf)
        if self._cache_budget > 0:
            nb = int(getattr(mat, "nbytes", 0))
            if self._cache_bytes + nb <= self._cache_budget:
                self._yz_cache[yz] = mat
                self._cache_bytes += nb
        return mat

    def _matrix_for_yz_f3raw(self, y: int, z: int) -> Any:
        """Return R[pq,vx] = <c|E_pq (Hdiag - epsa[v]) E_vx E_yz|c> for fixed (y,z)."""
        cp = self._cp
        y = int(y)
        z = int(z)
        if y < 0 or y >= int(self.norb) or z < 0 or z >= int(self.norb):
            raise ValueError("y/z out of range")
        yz = y * int(self.norb) + z
        cached = self._yz_cache_f3raw.get(yz)
        if cached is not None:
            return cached

        if self._use_tiled:
            mat = self._matrix_for_yz_f3raw_tiled(int(y), int(z))
        else:
            mat = self._matrix_for_yz_f3raw_full(int(y), int(z))
        if self._cache_budget > 0:
            nb = int(getattr(mat, "nbytes", 0))
            if self._cache_bytes + nb <= self._cache_budget:
                self._yz_cache_f3raw[yz] = mat
                self._cache_bytes += nb
        return mat

    def _matrix_for_yz_f3raw_full(self, y: int, z: int) -> Any:
        """Full-buffer yz-matrix build: one EPQ apply-all + one GEMM per (y,z)."""
        cp = self._cp

        yz = int(y) * int(self.norb) + int(z)
        yz_vec = self.t1_csf[:, yz]  # (ncsf,)

        ncsf = int(self.ncsf)
        norb = int(self.norb)
        nops = int(self.nops)

        # Reusable full buffer for K = E_all |yz_vec>.
        if self._k_csf is None or tuple(getattr(self._k_csf, "shape", ())) != (ncsf, nops):
            self._k_csf = cp.empty((ncsf, nops), dtype=cp.float64)

        if self._check_overflow:
            self._ws.overflow.fill(0)

        epq_apply_all_cuda_transpose_range(
            self.drt,
            self._ws,
            x=yz_vec,
            out=self._k_csf,
            k_start=0,
            k_count=ncsf,
            threads=int(self._apply_threads),
            sync=False,
            check_overflow=False,
        )

        # Scale k[:, v, x] by (hdiag - epsa[v]) for each v.
        k3 = self._k_csf.reshape(ncsf, norb, norb)
        scale = (self.hdiag[:, None] - self.epsa[None, :]).reshape(ncsf, norb, 1)
        k3 *= scale

        mat = cp.ascontiguousarray(self.bra_csf.T @ self._k_csf)

        if self._check_overflow:
            try:
                cp.cuda.get_current_stream().synchronize()
            except Exception:
                pass
            ov = int(cp.asnumpy(self._ws.overflow[0]))
            if ov != 0:
                raise RuntimeError(
                    f"EPQ apply overflow on GPU (overflow={ov}); reduce problem size or rebuild EPQ tables"
                )
        return mat

    def _matrix_for_yz_f3raw_tiled(self, y: int, z: int) -> Any:
        """Tile CSF dimension to avoid allocating a full (ncsf,nops) intermediate per (y,z)."""
        cp = self._cp

        yz = int(y) * int(self.norb) + int(z)
        yz_vec = self.t1_csf[:, yz]  # (ncsf,)

        ncsf = int(self.ncsf)
        norb = int(self.norb)
        nops = int(self.nops)
        tile_csf = int(self._tile_csf)

        # Reusable buffer for W-tile.
        if self._w_tile is None or tuple(getattr(self._w_tile, "shape", ())) != (tile_csf, nops):
            self._w_tile = cp.empty((tile_csf, nops), dtype=cp.float64)

        # Accumulate mat = sum_tiles bra_blk^T @ (scaled W_blk).
        mat = cp.zeros((nops, nops), dtype=cp.float64)

        # Best-effort overflow tracking (disabled by default for performance).
        if self._check_overflow:
            self._ws.overflow.fill(0)

        for k_start in range(0, ncsf, tile_csf):
            k_count = min(tile_csf, ncsf - int(k_start))
            w_tile = self._w_tile[:k_count]

            epq_apply_all_cuda_transpose_range(
                self.drt,
                self._ws,
                x=yz_vec,
                out=w_tile,
                k_start=int(k_start),
                k_count=int(k_count),
                threads=int(self._apply_threads),
                sync=False,
                check_overflow=False,
            )

            # Scale w_tile[:, v, x] by (hdiag - epsa[v]) for each v, in-place.
            w3 = w_tile.reshape(int(k_count), norb, norb)
            scale = (self.hdiag[int(k_start) : int(k_start) + int(k_count), None] - self.epsa[None, :]).reshape(
                int(k_count), norb, 1
            )
            w3 *= scale

            bra_blk = self.bra_csf[int(k_start) : int(k_start) + int(k_count), :]
            mat += bra_blk.T @ w_tile

        if self._check_overflow:
            try:
                cp.cuda.get_current_stream().synchronize()
            except Exception:
                pass
            ov = int(cp.asnumpy(self._ws.overflow[0]))
            if ov != 0:
                raise RuntimeError(
                    f"EPQ apply overflow on GPU (overflow={ov}); reduce problem size or rebuild EPQ tables"
                )

        return cp.ascontiguousarray(mat)

    # Convenience alias for callsites.
    def matrix_for_yz_f3raw(self, y: int, z: int) -> Any:
        return self._matrix_for_yz_f3raw(y, z)
