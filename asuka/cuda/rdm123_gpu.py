from __future__ import annotations

"""GPU RDM123 (dm1, dm2, dm3) builders for GUGA/CSF wavefunctions.

This module mirrors :func:`asuka.rdm.rdm123._make_rdm123_pyscf(..., reorder=False)`
and the OpenMolcas convention conversion in :func:`asuka.rdm.rdm123._reorder_dm123_molcas`,
but runs the dominant work on CUDA via CuPy + ASUKA's GUGA CUDA extension.

Key implementation choices
--------------------------
- Build the full `X[j,pq] = (E_pq|c>)[j]` table with the *destination-major* EPQ transpose
  tables (gather path) to avoid atomic scatter hot spots.
- For dm3, apply each generator `E_rs` to blocks of columns of `X` using the existing
  `epq_apply_gather_inplace_device` extension entrypoint (no custom RawKernels).
"""

from dataclasses import dataclass
import os
import time
from typing import Any

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuda.cuda_backend import (
    build_epq_action_table_transpose_device,
    build_epq_action_table_combined_device,
    build_occ_block_from_steps_inplace_device,
    build_w_diag_from_steps_inplace_device,
    build_w_from_epq_transpose_range_inplace_device,
    build_w_from_epq_transpose_range_mm_inplace_device,
    epq_apply_gather_inplace_device,
    has_build_w_from_epq_transpose_range_mm,
    has_cuda_ext,
    make_device_drt,
    make_device_state_cache,
)


def _env_int(name: str, default: int) -> int:
    import os

    v = os.getenv(name, None)
    if v is None:
        return int(default)
    try:
        return int(str(v).strip())
    except Exception:
        return int(default)


def _sync(cp) -> None:
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


@dataclass
class RDM123CudaWorkspace:
    norb: int
    ncsf: int
    nops: int

    drt_dev: Any
    state_dev: Any

    epq_table: Any | None = None
    epq_table_t: Any | None = None
    overflow: Any | None = None

    # Reusable buffers
    occ: Any | None = None  # (ncsf, norb)
    x_csf: Any | None = None  # (ncsf, nops)
    x2_csf: Any | None = None  # (ncsf, nops) second buffer (transition bra/ket)
    x_blk: Any | None = None  # (ncsf, nvec<=32)
    y_blk: Any | None = None  # (ncsf, nvec<=32)

    qp_perm: Any | None = None  # (nops,) int32

    # Reusable mm W-tile buffers keyed by out_cols (=nops*nvec)
    w_tile_mm: dict[int, Any] | None = None


def _get_or_make_workspace(drt: DRT) -> RDM123CudaWorkspace:
    ws = getattr(drt, "_rdm123_cuda_workspace", None)
    try:
        if (
            ws is not None
            and int(getattr(ws, "norb", -1)) == int(drt.norb)
            and int(getattr(ws, "ncsf", -1)) == int(drt.ncsf)
            and int(getattr(ws, "nops", -1)) == int(drt.norb) * int(drt.norb)
        ):
            return ws
    except Exception:
        pass

    drt_dev = make_device_drt(drt)
    state_dev = make_device_state_cache(drt, drt_dev)
    ws = RDM123CudaWorkspace(
        norb=int(drt.norb),
        ncsf=int(drt.ncsf),
        nops=int(drt.norb) * int(drt.norb),
        drt_dev=drt_dev,
        state_dev=state_dev,
    )
    ws.w_tile_mm = {}
    try:
        setattr(drt, "_rdm123_cuda_workspace", ws)
    except Exception:
        pass
    return ws


def _ensure_qp_perm(cp, ws: RDM123CudaWorkspace) -> Any:
    if ws.qp_perm is not None:
        return ws.qp_perm
    norb = int(ws.norb)
    nops = int(ws.nops)
    perm = np.arange(nops, dtype=np.int32).reshape(norb, norb).T.ravel()
    ws.qp_perm = cp.asarray(perm, dtype=cp.int32)
    return ws.qp_perm


def _ensure_epq_table(cp, drt: DRT, ws: RDM123CudaWorkspace, *, build_threads: int, profile: dict[str, float] | None) -> Any:
    if ws.epq_table is not None:
        return ws.epq_table
    cached = getattr(drt, "_epq_action_table_combined_device", None)
    if cached is not None:
        ws.epq_table = cached
        return ws.epq_table
    t0 = time.perf_counter()
    ws.epq_table = build_epq_action_table_combined_device(
        drt,
        ws.drt_dev,
        ws.state_dev,
        threads=int(build_threads),
        use_cache=True,
        sync=True,
        check_overflow=True,
    )
    if profile is not None:
        profile["rdm_epq_table_s"] = float(time.perf_counter() - t0)
    return ws.epq_table


def _ensure_epq_table_t(
    cp,
    drt: DRT,
    ws: RDM123CudaWorkspace,
    *,
    build_threads: int,
    profile: dict[str, float] | None,
) -> Any:
    if ws.epq_table_t is not None:
        return ws.epq_table_t
    if ws.epq_table is None:
        _ensure_epq_table(cp, drt, ws, build_threads=int(build_threads), profile=profile)

    t0 = time.perf_counter()
    ws.epq_table_t = build_epq_action_table_transpose_device(
        drt,
        ws.epq_table,
        use_cache=True,
        validate=False,
    )
    if profile is not None:
        profile["rdm_epq_table_t_s"] = float(time.perf_counter() - t0)
    return ws.epq_table_t


def _ensure_buffers(cp, ws: RDM123CudaWorkspace) -> None:
    ncsf = int(ws.ncsf)
    norb = int(ws.norb)
    nops = int(ws.nops)
    if ws.occ is None or tuple(getattr(ws.occ, "shape", ())) != (ncsf, norb):
        ws.occ = cp.empty((ncsf, norb), dtype=cp.float64)
    if ws.x_csf is None or tuple(getattr(ws.x_csf, "shape", ())) != (ncsf, nops):
        ws.x_csf = cp.empty((ncsf, nops), dtype=cp.float64)


def _ensure_buffers_trans(cp, ws: RDM123CudaWorkspace) -> None:
    """Ensure buffers needed for transition (bra/ket) RDM builds."""

    _ensure_buffers(cp, ws)
    ncsf = int(ws.ncsf)
    nops = int(ws.nops)
    if ws.x2_csf is None or tuple(getattr(ws.x2_csf, "shape", ())) != (ncsf, nops):
        ws.x2_csf = cp.empty((ncsf, nops), dtype=cp.float64)


def make_rdm123_raw_cuda(
    drt: DRT,
    ci_csf: Any,
    *,
    device: int | None = None,
    build_threads: int = 256,
    use_epq_table: bool = True,
    streaming_policy: str = "auto",
    profile: dict[str, float] | None = None,
) -> tuple[Any, Any, Any]:
    """Compute raw (dm1, dm2, dm3) on GPU in the same convention as `_make_rdm123_pyscf(reorder=False)`."""

    if not has_cuda_ext():
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")
    if not bool(use_epq_table):
        raise NotImplementedError("use_epq_table=False not implemented for RDM123 CUDA path")
    if str(streaming_policy).strip().lower() not in ("auto", "full"):
        raise ValueError("streaming_policy must be 'auto' or 'full'")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the CUDA RDM backend") from e

    if device is not None:
        cp.cuda.Device(int(device)).use()

    ws = _get_or_make_workspace(drt)
    _ensure_buffers(cp, ws)
    qp_perm = _ensure_qp_perm(cp, ws)

    norb = int(ws.norb)
    ncsf = int(ws.ncsf)
    nops = int(ws.nops)

    c = cp.ascontiguousarray(cp.asarray(ci_csf, dtype=cp.float64).ravel())
    if int(c.size) != ncsf:
        raise ValueError("ci_csf has wrong length")

    # EPQ table (cached on drt and/or workspace).
    epq_table = _ensure_epq_table(cp, drt, ws, build_threads=int(build_threads), profile=profile)
    epq_table_t = _ensure_epq_table_t(cp, drt, ws, build_threads=int(build_threads), profile=profile)
    if ws.overflow is None:
        ws.overflow = cp.empty((1,), dtype=cp.int32)
    else:
        ws.overflow = cp.asarray(ws.overflow, dtype=cp.int32).ravel()
        ws.overflow = cp.ascontiguousarray(ws.overflow)

    # Build occupancy table (used for diagonal E_rr applications).
    t0 = time.perf_counter()
    build_occ_block_from_steps_inplace_device(
        ws.state_dev,
        j_start=0,
        j_count=int(ncsf),
        occ_out=ws.occ,
        threads=256,
        sync=True,
    )
    if profile is not None:
        profile["rdm_build_occ_s"] = float(time.perf_counter() - t0)

    # Build X[i,pq] = (E_pq|c>)[i] in CSF-major layout (ncsf,nops).
    x = ws.x_csf
    stream = cp.cuda.get_current_stream()
    t0 = time.perf_counter()
    cp.cuda.runtime.memsetAsync(int(x.data.ptr), 0, int(x.size) * int(x.itemsize), int(stream.ptr))
    build_w_diag_from_steps_inplace_device(
        ws.state_dev,
        j_start=0,
        j_count=int(ncsf),
        x=c,
        w_out=x,
        threads=256,
        stream=stream,
        sync=False,
        relative_w=False,
    )
    ws.overflow.fill(0)
    build_w_from_epq_transpose_range_inplace_device(
        drt,
        ws.state_dev,
        epq_table_t,
        c,
        w_out=x,
        overflow=ws.overflow,
        threads=int(build_threads),
        stream=stream,
        sync=True,
        check_overflow=True,
        k_start=0,
        k_count=int(ncsf),
    )
    ov = int(cp.asnumpy(ws.overflow[0]))
    if ov != 0:
        raise RuntimeError(f"EPQ apply overflow on GPU (overflow={ov}); reduce problem size or rebuild EPQ tables")
    if profile is not None:
        profile["rdm_build_x_s"] = float(time.perf_counter() - t0)

    # dm1_raw[pq] = <c|E_pq|c> = X[:,pq]^T c
    t0 = time.perf_counter()
    dm1_pq = x.T @ c  # (nops,)
    dm1_raw = dm1_pq.reshape(norb, norb)

    # dm2_raw: dm2_flat[pq,rs] = sum_i X[i,qp] X[i,rs] = (X^T X)[qp,rs]
    gram = x.T @ x  # (nops,nops)
    dm2_flat = gram[qp_perm, :]
    dm2_raw = dm2_flat.reshape(norb, norb, norb, norb)
    if profile is not None:
        profile["rdm_dm1_dm2_s"] = float(time.perf_counter() - t0)

    # dm3_raw: build via either legacy gather-kernel path or yz-batched mm EPQ apply-all.
    dm3_flat = cp.empty((nops, nops, nops), dtype=cp.float64)

    impl = str(os.getenv("ASUKA_RDM123_DM3_IMPL", "auto")).strip().lower() or "auto"
    if impl not in ("auto", "legacy", "yz_mm"):
        raise ValueError("ASUKA_RDM123_DM3_IMPL must be one of: auto, legacy, yz_mm")

    use_yz_mm = bool(impl in ("auto", "yz_mm") and has_build_w_from_epq_transpose_range_mm())
    if impl == "yz_mm" and not use_yz_mm:
        # Safe fallback: keep correctness, but warn in verbose/profiling runs.
        if profile is not None:
            # Use a NaN marker so benches can notice the fallback without introducing non-float types.
            profile["rdm_dm3_yz_mm_unavailable_s"] = float("nan")

    stream_dm3 = cp.cuda.get_current_stream()
    t0 = time.perf_counter()

    dm3_apply_ms = 0.0
    dm3_gemm_ms = 0.0

    check_ov = bool(int(os.getenv("ASUKA_RDM123_DM3_CHECK_OVERFLOW", "0").strip() or "0"))

    if use_yz_mm:
        # yz batching is bounded by shared memory: nops*nvec*sizeof(double).
        yz_batch = _env_int("ASUKA_RDM123_DM3_YZ_BATCH", 8)
        yz_batch = max(1, yz_batch)
        max_smem_bytes = 48 * 1024
        max_batch = max(1, int(max_smem_bytes // (int(nops) * 8)))
        yz_batch = min(yz_batch, max_batch, 32, int(nops))

        tile_csf = _env_int("ASUKA_RDM123_DM3_TILE_CSF", 8192)
        tile_csf = max(1, min(int(tile_csf), int(ncsf)))

        # Reuse a W-tile buffer keyed by out_cols (=nops*nvec).
        if ws.w_tile_mm is None:
            ws.w_tile_mm = {}

        for yz0 in range(0, nops, yz_batch):
            yz1 = min(nops, int(yz0) + int(yz_batch))
            nvec = int(yz1 - yz0)
            out_cols = int(nops) * int(nvec)

            x_batch = x[:, int(yz0) : int(yz1)]  # (ncsf,nvec), pitched view allowed
            mat_batch_raw = cp.zeros((nops, out_cols), dtype=cp.float64)

            w_tile_full = ws.w_tile_mm.get(out_cols)
            if w_tile_full is None or tuple(getattr(w_tile_full, "shape", ())) != (tile_csf, out_cols):
                w_tile_full = cp.empty((tile_csf, out_cols), dtype=cp.float64)
                ws.w_tile_mm[out_cols] = w_tile_full

            # Record events only when profiling.
            ev_triples: list[tuple[Any, Any, Any]] = []
            for k_start in range(0, ncsf, tile_csf):
                k_count = min(tile_csf, ncsf - int(k_start))
                w_tile = w_tile_full[: int(k_count)]

                if check_ov:
                    ws.overflow.fill(0)

                ev0 = ev1 = ev2 = None
                if profile is not None:
                    ev0 = cp.cuda.Event()
                    ev1 = cp.cuda.Event()
                    ev2 = cp.cuda.Event()
                    ev0.record(stream_dm3)

                build_w_from_epq_transpose_range_mm_inplace_device(
                    drt,
                    ws.state_dev,
                    epq_table_t,
                    x_batch,
                    w_out=w_tile,
                    overflow=ws.overflow,
                    threads=int(build_threads),
                    stream=stream_dm3,
                    sync=False,
                    check_overflow=False,
                    k_start=int(k_start),
                    k_count=int(k_count),
                )

                if profile is not None and ev1 is not None:
                    ev1.record(stream_dm3)

                x_blk = x[int(k_start) : int(k_start) + int(k_count), :]
                mat_batch_raw += x_blk.T @ w_tile

                if profile is not None and ev2 is not None and ev0 is not None and ev1 is not None:
                    ev2.record(stream_dm3)
                    ev_triples.append((ev0, ev1, ev2))

            # Permute pq rows by qp_perm to match adjoint convention.
            mat_batch = mat_batch_raw[qp_perm, :]
            mat_batch_3 = mat_batch.reshape(nops, nops, nvec, order="C")
            dm3_flat[:, :, int(yz0) : int(yz1)] = mat_batch_3

            if profile is not None and ev_triples:
                try:
                    stream_dm3.synchronize()
                except Exception:
                    pass
                for ev0, ev1, ev2 in ev_triples:
                    dm3_apply_ms += float(cp.cuda.get_elapsed_time(ev0, ev1))
                    dm3_gemm_ms += float(cp.cuda.get_elapsed_time(ev1, ev2))

            if check_ov:
                try:
                    stream_dm3.synchronize()
                except Exception:
                    pass
                ov = int(cp.asnumpy(ws.overflow[0]))
                if ov != 0:
                    raise RuntimeError(
                        f"EPQ apply overflow on GPU (overflow={ov}); reduce problem size or rebuild EPQ tables"
                    )
    else:
        # Legacy path: loop over rs, apply E_rs to blocks of X columns using gather-kernels.
        nvec_max = min(32, int(nops))
        if ws.y_blk is None or tuple(getattr(ws.y_blk, "shape", ())) != (ncsf, nvec_max):
            ws.y_blk = cp.empty((ncsf, nvec_max), dtype=cp.float64)

        y_blk_full = ws.y_blk

        for rs in range(nops):
            r = rs // norb
            s = rs - r * norb
            for col0 in range(0, nops, nvec_max):
                col1 = min(nops, col0 + nvec_max)
                nvec = int(col1 - col0)
                x_view = x[:, int(col0) : int(col1)]
                y_view = y_blk_full[:, :nvec]

                if profile is not None:
                    ev0 = cp.cuda.Event()
                    ev1 = cp.cuda.Event()
                    ev2 = cp.cuda.Event()
                    ev0.record(stream_dm3)

                if r == s:
                    # Diagonal generator E_rr acts as occupancy scaling.
                    cp.multiply(x_view, ws.occ[:, r][:, None], out=y_view)
                else:
                    # Off-diagonal generator E_rs via destination-gather extension kernel.
                    if check_ov:
                        ws.overflow.fill(0)
                    epq_apply_gather_inplace_device(
                        drt,
                        ws.drt_dev,
                        ws.state_dev,
                        int(r),
                        int(s),
                        x_view,
                        y=y_view,
                        overflow=ws.overflow,
                        alpha=1.0,
                        threads=256,
                        add=False,
                        stream=stream_dm3,
                        sync=False,
                        check_overflow=False,
                    )

                if profile is not None:
                    ev1.record(stream_dm3)

                # H_block = X^T Y_block; then reorder pq rows by qp_perm.
                h_blk = x.T @ y_view  # (nops, nvec)
                dm3_flat[:, rs, int(col0) : int(col1)] = h_blk[qp_perm, :]

                if profile is not None:
                    ev2.record(stream_dm3)
                    try:
                        stream_dm3.synchronize()
                    except Exception:
                        pass
                    dm3_apply_ms += float(cp.cuda.get_elapsed_time(ev0, ev1))
                    dm3_gemm_ms += float(cp.cuda.get_elapsed_time(ev1, ev2))

                if check_ov and r != s:
                    try:
                        stream_dm3.synchronize()
                    except Exception:
                        pass
                    ov = int(cp.asnumpy(ws.overflow[0]))
                    if ov != 0:
                        raise RuntimeError(
                            f"EPQ apply overflow on GPU (overflow={ov}); reduce problem size or rebuild EPQ tables"
                        )

    if profile is not None:
        _sync(cp)
        profile["rdm_dm3_s"] = float(time.perf_counter() - t0)
        profile["rdm_dm3_apply_s"] = float(dm3_apply_ms * 1e-3)
        profile["rdm_dm3_gemm_s"] = float(dm3_gemm_ms * 1e-3)

    dm3_raw = dm3_flat.reshape(norb, norb, norb, norb, norb, norb)
    return dm1_raw, dm2_raw, dm3_raw


def reorder_dm123_molcas_cuda(
    dm1: Any,
    dm2: Any,
    dm3: Any,
    *,
    inplace: bool = True,
    profile: dict[str, float] | None = None,
) -> tuple[Any, Any, Any]:
    """Apply OpenMolcas-style delta corrections + symmetry to (dm2, dm3) on GPU."""

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for reorder_dm123_molcas_cuda") from e

    t0 = time.perf_counter()
    dm1 = cp.asarray(dm1, dtype=cp.float64)
    dm2 = cp.asarray(dm2, dtype=cp.float64)
    dm3 = cp.asarray(dm3, dtype=cp.float64)
    n = int(dm1.shape[0])
    if dm1.shape != (n, n):
        raise ValueError("dm1 must be square")
    if dm2.shape != (n, n, n, n):
        raise ValueError("dm2 must have shape (n,n,n,n)")
    if dm3.shape != (n, n, n, n, n, n):
        raise ValueError("dm3 must have shape (n,n,n,n,n,n)")

    if not bool(inplace):
        dm2 = dm2.copy()
        dm3 = dm3.copy()

    # 2-body delta correction:
    for k in range(n):
        dm2[:, k, k, :] -= dm1

    # MKFG3 diagonal symmetries (bra=ket):
    dm2 = cp.ascontiguousarray(
        0.25
        * (
            dm2
            + dm2.transpose(2, 3, 0, 1)
            + dm2.transpose(3, 2, 1, 0)
            + dm2.transpose(1, 0, 3, 2)
        ),
        dtype=cp.float64,
    )

    # 3-body delta correction:
    dm2_vxtz = dm2.transpose(2, 0, 1, 3)
    for q in range(n):
        dm3[:, q, q, :, :, :] -= dm2
        dm3[:, :, :, q, q, :] -= dm2
        dm3[:, q, :, :, q, :] -= dm2_vxtz
        for s in range(n):
            dm3[:, q, q, s, s, :] -= dm1

    # Full pair-permutation symmetry of TG3/G3:
    dm3_sym = cp.ascontiguousarray(dm3, dtype=cp.float64).copy()
    dm3_sym += dm3.transpose(2, 3, 0, 1, 4, 5)
    dm3_sym += dm3.transpose(4, 5, 2, 3, 0, 1)
    dm3_sym += dm3.transpose(0, 1, 4, 5, 2, 3)
    dm3_sym += dm3.transpose(2, 3, 4, 5, 0, 1)
    dm3_sym += dm3.transpose(4, 5, 0, 1, 2, 3)
    dm3 = cp.ascontiguousarray(dm3_sym * (1.0 / 6.0), dtype=cp.float64)

    if profile is not None:
        _sync(cp)
        profile["rdm_reorder_s"] = float(time.perf_counter() - t0)

    return cp.ascontiguousarray(dm1), dm2, dm3


def make_rdm123_molcas_cuda(
    drt: DRT,
    ci_csf: Any,
    *,
    device: int | None = None,
    build_threads: int = 256,
    use_epq_table: bool = True,
    streaming_policy: str = "auto",
    profile: dict[str, float] | None = None,
) -> tuple[Any, Any, Any]:
    """Compute (dm1, dm2, dm3) on GPU in OpenMolcas CASPT2 conventions."""

    dm1_raw, dm2_raw, dm3_raw = make_rdm123_raw_cuda(
        drt,
        ci_csf,
        device=device,
        build_threads=int(build_threads),
        use_epq_table=bool(use_epq_table),
        streaming_policy=str(streaming_policy),
        profile=profile,
    )
    return reorder_dm123_molcas_cuda(dm1_raw, dm2_raw, dm3_raw, inplace=True, profile=profile)


_P2LEV_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _p2lev_perm_inv(n: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(n)
    cached = _P2LEV_CACHE.get(n)
    if cached is not None:
        return cached
    from asuka.rdm.rdm123 import _molcas_p2lev_pairs  # noqa: PLC0415

    n2 = int(n * n)
    perm = np.asarray([il + n * jl for il, jl in _molcas_p2lev_pairs(n)], dtype=np.int64)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(n2, dtype=np.int64)
    _P2LEV_CACHE[n] = (perm, inv)
    return perm, inv


def make_trans_rdm123_raw_cuda(
    drt: DRT,
    ci_bra: Any,
    ci_ket: Any,
    *,
    device: int | None = None,
    build_threads: int = 256,
    use_epq_table: bool = True,
    streaming_policy: str = "auto",
    profile: dict[str, float] | None = None,
) -> tuple[Any, Any, Any]:
    """Compute raw transition (dm1, dm2, dm3) on GPU in the same convention as `_trans_rdm123_pyscf(..., reorder=False)`."""

    if not has_cuda_ext():
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")
    if not bool(use_epq_table):
        raise NotImplementedError("use_epq_table=False not implemented for transition RDM123 CUDA path")
    if str(streaming_policy).strip().lower() not in ("auto", "full"):
        raise ValueError("streaming_policy must be 'auto' or 'full'")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the CUDA RDM backend") from e

    if device is not None:
        cp.cuda.Device(int(device)).use()

    ws = _get_or_make_workspace(drt)
    _ensure_buffers_trans(cp, ws)
    qp_perm = _ensure_qp_perm(cp, ws)

    norb = int(ws.norb)
    ncsf = int(ws.ncsf)
    nops = int(ws.nops)

    cbra = cp.ascontiguousarray(cp.asarray(ci_bra, dtype=cp.float64).ravel())
    cket = cp.ascontiguousarray(cp.asarray(ci_ket, dtype=cp.float64).ravel())
    if int(cbra.size) != ncsf or int(cket.size) != ncsf:
        raise ValueError("ci_bra/ci_ket have wrong length")

    # EPQ tables (cached).
    epq_table_t = _ensure_epq_table_t(cp, drt, ws, build_threads=int(build_threads), profile=profile)
    if ws.overflow is None:
        ws.overflow = cp.empty((1,), dtype=cp.int32)
    else:
        ws.overflow = cp.asarray(ws.overflow, dtype=cp.int32).ravel()
        ws.overflow = cp.ascontiguousarray(ws.overflow)

    stream = cp.cuda.get_current_stream()

    # Build X_ket and X_bra in CSF-major layout: X[i,pq] = (E_pq|c>)[i]
    def _build_x(out, c, *, key: str) -> Any:
        t0 = time.perf_counter()
        cp.cuda.runtime.memsetAsync(int(out.data.ptr), 0, int(out.size) * int(out.itemsize), int(stream.ptr))
        build_w_diag_from_steps_inplace_device(
            ws.state_dev,
            j_start=0,
            j_count=int(ncsf),
            x=c,
            w_out=out,
            threads=256,
            stream=stream,
            sync=False,
            relative_w=False,
        )
        ws.overflow.fill(0)
        build_w_from_epq_transpose_range_inplace_device(
            drt,
            ws.state_dev,
            epq_table_t,
            c,
            w_out=out,
            overflow=ws.overflow,
            threads=int(build_threads),
            stream=stream,
            sync=True,
            check_overflow=True,
            k_start=0,
            k_count=int(ncsf),
        )
        ov = int(cp.asnumpy(ws.overflow[0]))
        if ov != 0:
            raise RuntimeError(
                f"EPQ apply overflow on GPU (overflow={ov}); reduce problem size or rebuild EPQ tables"
            )
        if profile is not None:
            profile[key] = float(time.perf_counter() - t0)
        return out

    x_ket = _build_x(ws.x_csf, cket, key="rdm_trans_build_x_ket_s")
    x_bra = _build_x(ws.x2_csf, cbra, key="rdm_trans_build_x_bra_s")

    # dm1_raw[pq] = <bra|E_pq|ket> = X_ket[:,pq]^T cbra
    t0 = time.perf_counter()
    dm1_pq = x_ket.T @ cbra
    dm1_raw = dm1_pq.reshape(norb, norb)

    # dm2_raw: dm2_flat[pq,rs] = <E_pq E_rs> with bra block E_qp|bra>
    gram_bk = x_bra.T @ x_ket  # rows: pq (bra), cols: rs (ket)
    dm2_flat = gram_bk[qp_perm, :]
    dm2_raw = dm2_flat.reshape(norb, norb, norb, norb)

    if profile is not None:
        profile["rdm_trans_dm1_dm2_s"] = float(time.perf_counter() - t0)

    # dm3_raw: prefer yz-batched mm EPQ apply-all (fast path), else fall back to legacy gather.
    dm3_flat = cp.empty((nops, nops, nops), dtype=cp.float64)
    t0 = time.perf_counter()
    dm3_apply_ms = 0.0
    dm3_gemm_ms = 0.0

    impl = str(os.getenv("ASUKA_RDM123_DM3_IMPL", "auto")).strip().lower() or "auto"
    if impl not in ("auto", "legacy", "yz_mm"):
        raise ValueError("ASUKA_RDM123_DM3_IMPL must be one of: auto, legacy, yz_mm")
    use_yz_mm = bool(impl in ("auto", "yz_mm") and has_build_w_from_epq_transpose_range_mm())
    if impl == "yz_mm" and not use_yz_mm and profile is not None:
        profile["rdm_trans_dm3_yz_mm_unavailable_s"] = float("nan")

    stream_dm3 = cp.cuda.get_current_stream()
    check_ov = bool(int(os.getenv("ASUKA_RDM123_DM3_CHECK_OVERFLOW", "0").strip() or "0"))

    if use_yz_mm:
        yz_batch = _env_int("ASUKA_RDM123_DM3_YZ_BATCH", 8)
        yz_batch = max(1, yz_batch)
        max_smem_bytes = 48 * 1024
        max_batch = max(1, int(max_smem_bytes // (int(nops) * 8)))
        yz_batch = min(yz_batch, max_batch, 32, int(nops))

        tile_csf = _env_int("ASUKA_RDM123_DM3_TILE_CSF", 8192)
        tile_csf = max(1, min(int(tile_csf), int(ncsf)))

        if ws.w_tile_mm is None:
            ws.w_tile_mm = {}

        for yz0 in range(0, nops, yz_batch):
            yz1 = min(nops, int(yz0) + int(yz_batch))
            nvec = int(yz1 - yz0)
            out_cols = int(nops) * int(nvec)

            x_batch = x_ket[:, int(yz0) : int(yz1)]
            mat_batch_raw = cp.zeros((nops, out_cols), dtype=cp.float64)

            w_tile_full = ws.w_tile_mm.get(out_cols)
            if w_tile_full is None or tuple(getattr(w_tile_full, "shape", ())) != (tile_csf, out_cols):
                w_tile_full = cp.empty((tile_csf, out_cols), dtype=cp.float64)
                ws.w_tile_mm[out_cols] = w_tile_full

            ev_triples: list[tuple[Any, Any, Any]] = []
            for k_start in range(0, ncsf, tile_csf):
                k_count = min(tile_csf, ncsf - int(k_start))
                w_tile = w_tile_full[: int(k_count)]

                if check_ov:
                    ws.overflow.fill(0)

                ev0 = ev1 = ev2 = None
                if profile is not None:
                    ev0 = cp.cuda.Event()
                    ev1 = cp.cuda.Event()
                    ev2 = cp.cuda.Event()
                    ev0.record(stream_dm3)

                build_w_from_epq_transpose_range_mm_inplace_device(
                    drt,
                    ws.state_dev,
                    epq_table_t,
                    x_batch,
                    w_out=w_tile,
                    overflow=ws.overflow,
                    threads=int(build_threads),
                    stream=stream_dm3,
                    sync=False,
                    check_overflow=False,
                    k_start=int(k_start),
                    k_count=int(k_count),
                )

                if profile is not None and ev1 is not None:
                    ev1.record(stream_dm3)

                x_bra_blk = x_bra[int(k_start) : int(k_start) + int(k_count), :]
                mat_batch_raw += x_bra_blk.T @ w_tile

                if profile is not None and ev2 is not None and ev0 is not None and ev1 is not None:
                    ev2.record(stream_dm3)
                    ev_triples.append((ev0, ev1, ev2))

            mat_batch = mat_batch_raw[qp_perm, :]
            mat_batch_3 = mat_batch.reshape(nops, nops, nvec, order="C")
            dm3_flat[:, :, int(yz0) : int(yz1)] = mat_batch_3

            if profile is not None and ev_triples:
                try:
                    stream_dm3.synchronize()
                except Exception:
                    pass
                for ev0, ev1, ev2 in ev_triples:
                    dm3_apply_ms += float(cp.cuda.get_elapsed_time(ev0, ev1))
                    dm3_gemm_ms += float(cp.cuda.get_elapsed_time(ev1, ev2))

            if check_ov:
                try:
                    stream_dm3.synchronize()
                except Exception:
                    pass
                ov = int(cp.asnumpy(ws.overflow[0]))
                if ov != 0:
                    raise RuntimeError(
                        f"EPQ apply overflow on GPU (overflow={ov}); reduce problem size or rebuild EPQ tables"
                    )
    else:
        # Legacy path: loop over rs, apply E_rs to blocks of X_ket columns using gather-kernels.
        # (Kept for fallback parity if mm builder is unavailable.)
        if ws.occ is None or tuple(getattr(ws.occ, "shape", ())) != (ncsf, norb):
            ws.occ = cp.empty((ncsf, norb), dtype=cp.float64)
        if ws.y_blk is None or tuple(getattr(ws.y_blk, "shape", ())) != (ncsf, min(32, int(nops))):
            ws.y_blk = cp.empty((ncsf, min(32, int(nops))), dtype=cp.float64)

        # Occupancy table (used for diagonal E_rr).
        build_occ_block_from_steps_inplace_device(
            ws.state_dev,
            j_start=0,
            j_count=int(ncsf),
            occ_out=ws.occ,
            threads=256,
            sync=True,
        )

        nvec_max = min(32, int(nops))
        y_blk_full = ws.y_blk
        for rs in range(nops):
            r = rs // norb
            s = rs - r * norb
            for col0 in range(0, nops, nvec_max):
                col1 = min(nops, col0 + nvec_max)
                nvec = int(col1 - col0)
                x_view = x_ket[:, int(col0) : int(col1)]
                y_view = y_blk_full[:, :nvec]

                if profile is not None:
                    ev0 = cp.cuda.Event()
                    ev1 = cp.cuda.Event()
                    ev2 = cp.cuda.Event()
                    ev0.record(stream_dm3)

                if r == s:
                    cp.multiply(x_view, ws.occ[:, r][:, None], out=y_view)
                else:
                    if check_ov:
                        ws.overflow.fill(0)
                    epq_apply_gather_inplace_device(
                        drt,
                        ws.drt_dev,
                        ws.state_dev,
                        int(r),
                        int(s),
                        x_view,
                        y=y_view,
                        overflow=ws.overflow,
                        alpha=1.0,
                        threads=256,
                        add=False,
                        stream=stream_dm3,
                        sync=False,
                        check_overflow=False,
                    )

                if profile is not None:
                    ev1.record(stream_dm3)

                h_blk = x_bra.T @ y_view
                dm3_flat[:, rs, int(col0) : int(col1)] = h_blk[qp_perm, :]

                if profile is not None:
                    ev2.record(stream_dm3)
                    try:
                        stream_dm3.synchronize()
                    except Exception:
                        pass
                    dm3_apply_ms += float(cp.cuda.get_elapsed_time(ev0, ev1))
                    dm3_gemm_ms += float(cp.cuda.get_elapsed_time(ev1, ev2))

                if check_ov and r != s:
                    try:
                        stream_dm3.synchronize()
                    except Exception:
                        pass
                    ov = int(cp.asnumpy(ws.overflow[0]))
                    if ov != 0:
                        raise RuntimeError(
                            f"EPQ apply overflow on GPU (overflow={ov}); reduce problem size or rebuild EPQ tables"
                        )

    if profile is not None:
        _sync(cp)
        profile["rdm_trans_dm3_s"] = float(time.perf_counter() - t0)
        profile["rdm_trans_dm3_apply_s"] = float(dm3_apply_ms * 1e-3)
        profile["rdm_trans_dm3_gemm_s"] = float(dm3_gemm_ms * 1e-3)

    dm3_raw = dm3_flat.reshape(norb, norb, norb, norb, norb, norb)
    return dm1_raw, dm2_raw, dm3_raw


def reorder_dm123_molcas_trans_cuda(
    dm1: Any,
    dm2: Any,
    dm3: Any,
    *,
    inplace: bool = True,
    profile: dict[str, float] | None = None,
) -> tuple[Any, Any, Any]:
    """Apply OpenMolcas-style transition TG1/TG2/TG3 reorder to (dm1, dm2, dm3) on GPU."""

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for reorder_dm123_molcas_trans_cuda") from e

    t0 = time.perf_counter()
    dm1 = cp.ascontiguousarray(cp.asarray(dm1, dtype=cp.float64))
    dm2 = cp.ascontiguousarray(cp.asarray(dm2, dtype=cp.float64))
    dm3 = cp.ascontiguousarray(cp.asarray(dm3, dtype=cp.float64))
    n = int(dm1.shape[0])
    if dm1.shape != (n, n):
        raise ValueError("dm1 must be square")
    if dm2.shape != (n, n, n, n):
        raise ValueError("dm2 must have shape (n,n,n,n)")
    if dm3.shape != (n, n, n, n, n, n):
        raise ValueError("dm3 must have shape (n,n,n,n,n,n)")

    if not bool(inplace):
        dm2 = dm2.copy()
        dm3 = dm3.copy()

    # 2-body delta correction (transition; no bra=ket hermitization):
    for k in range(n):
        dm2[:, k, k, :] -= dm1

    # Enforce OpenMolcas P2LEV canonical-half symmetry for TG2 by copying.
    n2 = int(n * n)
    perm_h, inv_h = _p2lev_perm_inv(n)
    perm = cp.asarray(perm_h, dtype=cp.int64)
    inv = cp.asarray(inv_h, dtype=cp.int64)

    m = dm2.reshape(n2, n2, order="F")
    m_p = m[perm][:, perm]
    m_p_lo = cp.tril(m_p)
    m_p_sym = m_p_lo + cp.tril(m_p, -1).T
    m_sym = m_p_sym[inv][:, inv]
    dm2 = cp.ascontiguousarray(m_sym.reshape(n, n, n, n, order="F"), dtype=cp.float64)

    # 3-body delta correction (transition; no bra=ket symmetrization):
    dm2_vxtz = dm2.transpose(2, 0, 1, 3)  # (t,v,x,z) = dm2[v,x,t,z]
    for q in range(n):
        dm3[:, q, q, :, :, :] -= dm2
        dm3[:, :, :, q, q, :] -= dm2
        dm3[:, q, :, :, q, :] -= dm2_vxtz
        for s in range(n):
            dm3[:, q, q, s, s, :] -= dm1

    # Full pair-permutation symmetry of TG3:
    dm3_sym = cp.ascontiguousarray(dm3, dtype=cp.float64).copy()
    dm3_sym += dm3.transpose(2, 3, 0, 1, 4, 5)
    dm3_sym += dm3.transpose(4, 5, 2, 3, 0, 1)
    dm3_sym += dm3.transpose(0, 1, 4, 5, 2, 3)
    dm3_sym += dm3.transpose(2, 3, 4, 5, 0, 1)
    dm3_sym += dm3.transpose(4, 5, 0, 1, 2, 3)
    dm3 = cp.ascontiguousarray(dm3_sym * (1.0 / 6.0), dtype=cp.float64)

    if profile is not None:
        _sync(cp)
        profile["rdm_trans_reorder_s"] = float(time.perf_counter() - t0)

    return cp.ascontiguousarray(dm1), dm2, dm3


def make_trans_rdm123_molcas_cuda(
    drt: DRT,
    ci_bra: Any,
    ci_ket: Any,
    *,
    device: int | None = None,
    build_threads: int = 256,
    use_epq_table: bool = True,
    streaming_policy: str = "auto",
    profile: dict[str, float] | None = None,
) -> tuple[Any, Any, Any]:
    """Compute transition (tg1, tg2, tg3) on GPU in OpenMolcas CASPT2 conventions."""

    dm1_raw, dm2_raw, dm3_raw = make_trans_rdm123_raw_cuda(
        drt,
        ci_bra,
        ci_ket,
        device=device,
        build_threads=int(build_threads),
        use_epq_table=bool(use_epq_table),
        streaming_policy=str(streaming_policy),
        profile=profile,
    )
    return reorder_dm123_molcas_trans_cuda(dm1_raw, dm2_raw, dm3_raw, inplace=True, profile=profile)
