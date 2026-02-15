from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.cuda.cublas_workspace import recommend_cublas_workspace_bytes_for_emulated_fp64_gemm
from asuka.cuda.cuda_backend import (
    build_epq_action_table_combined_device,
    build_t_block_epq_atomic_inplace_device,
    build_w_diag_from_steps_inplace_device,
    build_w_from_epq_table_inplace_device,
    has_cuda_ext,
    make_device_drt,
    make_device_state_cache,
    make_rdm_gram_workspace,
    rdm_cross_gram_and_dm1_inplace_device,
    rdm_cross_gram_and_dm1_csf_major_inplace_device,
    rdm_gram_and_dm1_inplace_device,
    rdm_gram_and_dm1_csf_major_inplace_device,
)
from asuka.cuguga.drt import DRT


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, None)
    if v is None:
        return int(default)
    try:
        return int(str(v).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, None)
    if v is None:
        return float(default)
    try:
        return float(str(v).strip())
    except Exception:
        return float(default)


def _choose_tiling_csf(
    cp,
    *,
    ncsf: int,
    nops: int,
    full_buffer_factor: int = 1,
    streaming_ncsf_cutoff: int | None = None,
) -> tuple[bool, int]:
    """Choose whether to build T in CSF tiles, and the tile size."""

    tile_csf_env = int(_env_int("CUGUGA_RDM_CUDA_TILE_CSF", 0))
    mem_fraction = float(_env_float("CUGUGA_RDM_CUDA_TILE_MEM_FRACTION", 0.6))
    tile_min_bytes = int(_env_int("CUGUGA_RDM_CUDA_TILE_MIN_BYTES", 512 * 1024 * 1024))
    tile_fallback_csf = int(_env_int("CUGUGA_RDM_CUDA_STREAMING_TILE_FALLBACK_CSF", 8192))
    if not (0.0 < mem_fraction <= 1.0):
        mem_fraction = 0.6
    tile_fallback_csf = max(1, int(tile_fallback_csf))

    ncsf = int(ncsf)
    nops = int(nops)
    full_buffer_factor = max(1, int(full_buffer_factor))
    if streaming_ncsf_cutoff is None:
        streaming_ncsf_cutoff = int(_env_int("CUGUGA_RDM_CUDA_STREAMING_NCSF_CUTOFF", 2_000_000))
    else:
        streaming_ncsf_cutoff = int(streaming_ncsf_cutoff)
    if streaming_ncsf_cutoff < 0:
        streaming_ncsf_cutoff = 0
    if ncsf <= 0 or nops <= 0:
        return False, 1

    if tile_csf_env > 0:
        tile_csf = max(1, min(int(tile_csf_env), ncsf))
        return bool(tile_csf < ncsf), int(tile_csf)

    full_bytes = int(full_buffer_factor) * int(ncsf) * int(nops) * int(np.dtype(np.float64).itemsize)
    try:
        free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
        free_bytes = int(free_bytes)
    except Exception:
        free_bytes = 0

    force_streaming = bool(streaming_ncsf_cutoff > 0 and int(ncsf) >= int(streaming_ncsf_cutoff))
    use_tiled = bool(force_streaming)
    tile_csf = int(ncsf)
    threshold_bytes = int(tile_min_bytes) * max(1, int(full_buffer_factor))
    if free_bytes > 0 and (
        full_bytes >= int(threshold_bytes) or full_bytes >= int(float(mem_fraction) * float(free_bytes))
    ):
        use_tiled = True
        budget = int(float(free_bytes) * float(mem_fraction))
        denom = int(nops) * int(np.dtype(np.float64).itemsize)
        tile_csf = int(budget // denom) if denom > 0 else 0

    if use_tiled and tile_csf >= int(ncsf) and force_streaming:
        tile_csf = int(min(int(ncsf), int(tile_fallback_csf)))

    tile_csf = max(1, min(int(tile_csf), ncsf))
    return bool(use_tiled and tile_csf < ncsf), int(tile_csf)


@dataclass
class RDM12CudaWorkspace:
    """Reusable device-side workspace for RDM12 evaluation."""

    norb: int
    ncsf: int
    nops: int

    drt_dev: Any
    state_dev: Any

    p_gpu: Any
    q_gpu: Any

    overflow: Any
    gram_ws: Any

    # Optional reusable buffers (allocated lazily)
    dm1_pq_gpu: Any | None = None
    gram0_gpu: Any | None = None
    t_tile: Any | None = None
    t_tile_aux: Any | None = None
    t_tile_b: Any | None = None  # second buffer for double-buffered pipeline
    t_tile_aux_b: Any | None = None  # second aux buffer for double-buffered transition RDM

    def configure_gram_ws(
        self,
        *,
        gemm_backend: str,
        math_mode: str,
        cublas_workspace_mb: int = 0,
        emulation_strategy: str | None = None,
        fixed_point_mantissa_control: str | None = None,
        fixed_point_max_mantissa_bits: int | None = None,
        fixed_point_mantissa_bit_offset: int | None = None,
    ) -> None:
        self.gram_ws.set_gemm_backend(str(gemm_backend))
        self.gram_ws.set_cublas_math_mode(str(math_mode))
        if emulation_strategy is not None:
            self.gram_ws.set_cublas_emulation_strategy(str(emulation_strategy))
        if fixed_point_mantissa_control is not None:
            self.gram_ws.set_cublas_fixed_point_mantissa_control(str(fixed_point_mantissa_control))
        if fixed_point_max_mantissa_bits is not None:
            self.gram_ws.set_cublas_fixed_point_max_mantissa_bits(int(fixed_point_max_mantissa_bits))
        if fixed_point_mantissa_bit_offset is not None:
            self.gram_ws.set_cublas_fixed_point_mantissa_bit_offset(int(fixed_point_mantissa_bit_offset))

        # Workspace policy:
        # - If user provides a positive workspace size, respect it.
        # - Otherwise, if emulated-FP64 GEMMEx is enabled, allocate a conservative "safe bound"
        #   workspace to avoid slow internal allocations and enable CUDA Graph capture stability.
        ws_mb = int(cublas_workspace_mb)
        if ws_mb > 0:
            self.gram_ws.set_cublas_workspace_bytes(ws_mb * 1024 * 1024)
            return

        # Keep legacy behavior for native FP64: default to 0 workspace.
        ws_info = dict(self.gram_ws.cublas_emulation_info())
        cap_bytes = 2048 * 1024 * 1024  # safety cap; override by passing an explicit positive MB value
        rec = recommend_cublas_workspace_bytes_for_emulated_fp64_gemm(
            ws_info=ws_info,
            gemm_shapes=[(int(self.nops), int(self.nops), int(self.ncsf))],
            batch_count=1,
            is_complex=False,
            cap_bytes=int(cap_bytes),
        )
        self.gram_ws.set_cublas_workspace_bytes(int(rec))



def make_rdm12_cuda_workspace(
    drt: DRT,
    *,
    gemm_backend: str = "gemmex_fp64",
    math_mode: str = "default",
    cublas_workspace_mb: int = 0,
    emulation_strategy: str | None = None,
    fixed_point_mantissa_control: str | None = None,
    fixed_point_max_mantissa_bits: int | None = None,
    fixed_point_mantissa_bit_offset: int | None = None,
) -> RDM12CudaWorkspace:
    """Create a reusable device-side workspace for `make_rdm12_cuda`."""

    norb = int(drt.norb)
    ncsf = int(drt.ncsf)
    if norb <= 0:
        raise ValueError("invalid norb")
    if ncsf <= 0:
        raise ValueError("invalid ncsf")

    if not has_cuda_ext():
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the CUDA RDM backend") from e

    nops = norb * norb
    p_host = np.repeat(np.arange(norb, dtype=np.int32), norb)
    q_host = np.tile(np.arange(norb, dtype=np.int32), norb)

    drt_dev = make_device_drt(drt)
    state_dev = make_device_state_cache(drt, drt_dev)

    p_gpu = cp.asarray(p_host, dtype=cp.int32)
    q_gpu = cp.asarray(q_host, dtype=cp.int32)
    overflow = cp.empty((1,), dtype=cp.int32)

    gram_ws = make_rdm_gram_workspace(nops=nops)
    ws = RDM12CudaWorkspace(
        norb=norb,
        ncsf=ncsf,
        nops=nops,
        drt_dev=drt_dev,
        state_dev=state_dev,
        p_gpu=p_gpu,
        q_gpu=q_gpu,
        overflow=overflow,
        gram_ws=gram_ws,
    )
    ws.configure_gram_ws(
        gemm_backend=str(gemm_backend),
        math_mode=str(math_mode),
        cublas_workspace_mb=int(cublas_workspace_mb),
        emulation_strategy=emulation_strategy,
        fixed_point_mantissa_control=fixed_point_mantissa_control,
        fixed_point_max_mantissa_bits=fixed_point_max_mantissa_bits,
        fixed_point_mantissa_bit_offset=fixed_point_mantissa_bit_offset,
    )
    return ws


def make_rdm12_cuda(
    drt: DRT,
    civec: np.ndarray,
    *,
    block_nops: int = 8,
    build_threads: int = 256,
    workspace: RDM12CudaWorkspace | None = None,
    use_epq_table: bool | None = None,
    gemm_backend: str = "gemmex_fp64",
    math_mode: str = "default",
    cublas_workspace_mb: int = 0,
    emulation_strategy: str | None = None,
    fixed_point_mantissa_control: str | None = None,
    fixed_point_max_mantissa_bits: int | None = None,
    fixed_point_mantissa_bit_offset: int | None = None,
    symmetrize_gram: bool = True,
    streaming_ncsf_cutoff: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (dm1, dm2) on GPU using (segment-walk or epq_table -> T) then cuBLAS (dm1, Gram).

    Notes
    -----
    - `T[pq,:] = (E_pq |c>)` is built on GPU with FP64 atomics.
    - `dm1[q,p] = dot(c, T[pq,:])`.
    - `Gram0[pq,rs] = dot(T[pq,:], T[rs,:])`, then reordered to `Gram[pq,rs] = Gram0[qp,rs]`
      to match the convention in `asuka.rdm.stream.make_rdm12_streaming`.
    - `Gram0` should be symmetric; if atomic summation introduces tiny asymmetry, set `symmetrize_gram=True`.
    """

    norb = int(drt.norb)
    ncsf = int(drt.ncsf)
    if norb <= 0:
        raise ValueError("invalid norb")
    if ncsf <= 0:
        raise ValueError("invalid ncsf")

    if not has_cuda_ext():
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the CUDA RDM backend") from e

    c = np.asarray(civec, dtype=np.float64).ravel()
    if c.size != ncsf:
        raise ValueError("civec has wrong length")

    cache_workspace = bool(_env_int("CUGUGA_RDM_CUDA_CACHE_WORKSPACE", 1))
    if workspace is None and cache_workspace:
        ws_cached = getattr(drt, "_rdm12_cuda_workspace", None)
        if ws_cached is not None:
            try:
                if (
                    int(getattr(ws_cached, "norb", -1)) == norb
                    and int(getattr(ws_cached, "ncsf", -1)) == ncsf
                    and int(getattr(ws_cached, "nops", -1)) == norb * norb
                ):
                    workspace = ws_cached
            except Exception:
                pass

    nops = norb * norb
    block_nops = int(block_nops)
    if block_nops < 1:
        raise ValueError("block_nops must be >= 1")
    block_nops = min(block_nops, nops)

    if workspace is None:
        workspace = make_rdm12_cuda_workspace(
            drt,
            gemm_backend=str(gemm_backend),
            math_mode=str(math_mode),
            cublas_workspace_mb=int(cublas_workspace_mb),
            emulation_strategy=emulation_strategy,
            fixed_point_mantissa_control=fixed_point_mantissa_control,
            fixed_point_max_mantissa_bits=fixed_point_max_mantissa_bits,
            fixed_point_mantissa_bit_offset=fixed_point_mantissa_bit_offset,
        )
        if cache_workspace:
            try:
                setattr(drt, "_rdm12_cuda_workspace", workspace)
            except Exception:
                pass
    else:
        if int(workspace.norb) != norb or int(workspace.ncsf) != ncsf or int(workspace.nops) != nops:
            raise ValueError("workspace does not match (norb,ncsf)")
        workspace.configure_gram_ws(
            gemm_backend=str(gemm_backend),
            math_mode=str(math_mode),
            cublas_workspace_mb=int(cublas_workspace_mb),
            emulation_strategy=emulation_strategy,
            fixed_point_mantissa_control=fixed_point_mantissa_control,
            fixed_point_max_mantissa_bits=fixed_point_max_mantissa_bits,
            fixed_point_mantissa_bit_offset=fixed_point_mantissa_bit_offset,
        )

    c_gpu = cp.asarray(c, dtype=cp.float64)
    drt_dev = workspace.drt_dev
    state_dev = workspace.state_dev
    p_gpu = workspace.p_gpu
    q_gpu = workspace.q_gpu

    overflow = workspace.overflow
    epq_table = None
    if use_epq_table is not False:
        if use_epq_table is True:
            epq_table = build_epq_action_table_combined_device(
                drt,
                drt_dev,
                state_dev,
                threads=int(build_threads),
                sync=True,
                check_overflow=True,
                use_cache=True,
            )
        else:
            epq_table = getattr(drt, "_epq_action_table_combined_device", None)

    if epq_table is not None:
        use_tiled, tile_csf = _choose_tiling_csf(
            cp,
            ncsf=int(ncsf),
            nops=int(nops),
            full_buffer_factor=1,
            streaming_ncsf_cutoff=streaming_ncsf_cutoff,
        )

        if use_tiled:
            stream = cp.cuda.get_current_stream()

            if workspace.t_tile is None or getattr(workspace.t_tile, "shape", None) != (tile_csf, nops):
                workspace.t_tile = cp.empty((tile_csf, nops), dtype=cp.float64)
            if workspace.t_tile_b is None or getattr(workspace.t_tile_b, "shape", None) != (tile_csf, nops):
                workspace.t_tile_b = cp.empty((tile_csf, nops), dtype=cp.float64)
            if workspace.dm1_pq_gpu is None or getattr(workspace.dm1_pq_gpu, "shape", None) != (nops,):
                workspace.dm1_pq_gpu = cp.empty((nops,), dtype=cp.float64)
            if workspace.gram0_gpu is None or getattr(workspace.gram0_gpu, "shape", None) != (nops, nops):
                workspace.gram0_gpu = cp.empty((nops, nops), dtype=cp.float64)

            dm1_pq_gpu = workspace.dm1_pq_gpu
            gram0_gpu = workspace.gram0_gpu
            dm1_pq_gpu.fill(0)
            gram0_gpu.fill(0)

            t_bufs = [workspace.t_tile, workspace.t_tile_b]
            ntiles = (ncsf + tile_csf - 1) // tile_csf

            if ntiles >= 2:
                # Double-buffered pipeline: overlap T-build with GEMM contraction.
                stream_build = cp.cuda.Stream(non_blocking=True)
                stream_contract = cp.cuda.Stream(non_blocking=True)
                # Ensure the new streams wait for any prior work on the default stream.
                init_event = stream.record()
                stream_build.wait_event(init_event)
                stream_contract.wait_event(init_event)
                event_built = [cp.cuda.Event() for _ in range(2)]
                event_contracted = [cp.cuda.Event() for _ in range(2)]

                for tile_idx in range(ntiles):
                    buf_idx = tile_idx % 2
                    k_start = tile_idx * tile_csf
                    k_count = min(tile_csf, ncsf - k_start)
                    t_tile = t_bufs[buf_idx][:k_count]

                    # Wait until previous contraction of THIS buffer is done before overwriting.
                    if tile_idx >= 2:
                        stream_build.wait_event(event_contracted[buf_idx])

                    # Build T-tile on the build stream.
                    cp.cuda.runtime.memsetAsync(
                        int(t_tile.data.ptr), 0,
                        int(t_tile.size) * int(t_tile.itemsize),
                        int(stream_build.ptr),
                    )
                    build_w_diag_from_steps_inplace_device(
                        state_dev,
                        j_start=int(k_start),
                        j_count=int(k_count),
                        x=c_gpu,
                        w_out=t_tile,
                        threads=256,
                        stream=stream_build,
                        sync=False,
                        relative_w=True,
                    )
                    build_w_from_epq_table_inplace_device(
                        drt,
                        state_dev,
                        epq_table,
                        c_gpu,
                        w_out=t_tile,
                        overflow=overflow,
                        threads=int(build_threads),
                        stream=stream_build,
                        sync=False,
                        check_overflow=False,
                        k_start=int(k_start),
                        k_count=int(k_count),
                    )
                    event_built[buf_idx] = stream_build.record()

                    # Contract T-tile on the contract stream (wait for build to finish first).
                    stream_contract.wait_event(event_built[buf_idx])
                    c_tile = c_gpu[k_start : k_start + k_count]
                    rdm_gram_and_dm1_csf_major_inplace_device(
                        workspace.gram_ws,
                        t_tile,
                        c_tile,
                        dm1_out=dm1_pq_gpu,
                        gram_out=gram0_gpu,
                        stream=stream_contract,
                        sync=False,
                        accumulate=True,
                    )
                    event_contracted[buf_idx] = stream_contract.record()

                # Make the default stream wait for the pipeline to finish.
                stream.wait_event(event_contracted[(ntiles - 1) % 2])
            else:
                # Single tile: no pipeline needed.
                k_count = min(tile_csf, ncsf)
                t_tile = workspace.t_tile[:k_count]

                cp.cuda.runtime.memsetAsync(
                    int(t_tile.data.ptr), 0,
                    int(t_tile.size) * int(t_tile.itemsize),
                    int(stream.ptr),
                )
                build_w_diag_from_steps_inplace_device(
                    state_dev,
                    j_start=0,
                    j_count=int(k_count),
                    x=c_gpu,
                    w_out=t_tile,
                    threads=256,
                    stream=stream,
                    sync=False,
                    relative_w=True,
                )
                build_w_from_epq_table_inplace_device(
                    drt,
                    state_dev,
                    epq_table,
                    c_gpu,
                    w_out=t_tile,
                    overflow=overflow,
                    threads=int(build_threads),
                    stream=stream,
                    sync=False,
                    check_overflow=False,
                    k_start=0,
                    k_count=int(k_count),
                )

                c_tile = c_gpu[:k_count]
                rdm_gram_and_dm1_csf_major_inplace_device(
                    workspace.gram_ws,
                    t_tile,
                    c_tile,
                    dm1_out=dm1_pq_gpu,
                    gram_out=gram0_gpu,
                    stream=stream,
                    sync=False,
                    accumulate=True,
                )

        else:
            # CSF-major build: T_csf[i,pq] = (E_pq|c>)[i].
            # This avoids the extreme atomic contention of building the operator-major
            # layout (pq-major) directly for large ncsf.
            stream = cp.cuda.get_current_stream()
            t_csf = cp.empty((ncsf, nops), dtype=cp.float64)
            cp.cuda.runtime.memsetAsync(int(t_csf.data.ptr), 0, int(t_csf.size) * int(t_csf.itemsize), int(stream.ptr))

            build_w_diag_from_steps_inplace_device(
                state_dev,
                j_start=0,
                j_count=int(ncsf),
                x=c_gpu,
                w_out=t_csf,
                threads=256,
                stream=stream,
                sync=False,
            )
            build_w_from_epq_table_inplace_device(
                drt,
                state_dev,
                epq_table,
                c_gpu,
                w_out=t_csf,
                overflow=overflow,
                threads=int(build_threads),
                stream=stream,
                sync=False,
                check_overflow=False,
            )
            dm1_pq_gpu, gram0_gpu = rdm_gram_and_dm1_csf_major_inplace_device(workspace.gram_ws, t_csf, c_gpu)
    else:
        T_gpu = cp.empty((nops, ncsf), dtype=cp.float64)
        for i0 in range(0, nops, block_nops):
            i1 = min(nops, i0 + block_nops)
            out_view = T_gpu[i0:i1]
            out_view, overflow = build_t_block_epq_atomic_inplace_device(
                drt,
                drt_dev,
                state_dev,
                c_gpu,
                p_gpu[i0:i1],
                q_gpu[i0:i1],
                out=out_view,
                overflow=overflow,
                threads=int(build_threads),
                zero_out=True,
                sync=True,
                check_overflow=True,
            )
        dm1_pq_gpu, gram0_gpu = rdm_gram_and_dm1_inplace_device(workspace.gram_ws, T_gpu, c_gpu)
    dm1_pq = cp.asnumpy(dm1_pq_gpu)
    gram0 = cp.asnumpy(gram0_gpu).T.copy()

    if symmetrize_gram:
        gram0 = 0.5 * (gram0 + gram0.T)

    dm1 = dm1_pq.reshape(norb, norb).T.copy()

    swap = np.arange(nops, dtype=np.int32).reshape(norb, norb).T.ravel()
    gram = gram0[swap]

    dm2 = gram.reshape(norb, norb, norb, norb)
    for p in range(norb):
        for q in range(norb):
            dm2[p, q, q, :] -= dm1[:, p]

    return np.asarray(dm1, dtype=np.float64), np.asarray(dm2, dtype=np.float64)


def trans_rdm12_cuda(
    drt: DRT,
    ci_bra: np.ndarray,
    ci_ket: np.ndarray,
    *,
    block_nops: int = 8,
    build_threads: int = 256,
    workspace: RDM12CudaWorkspace | None = None,
    use_epq_table: bool | None = None,
    gemm_backend: str = "gemmex_fp64",
    math_mode: str = "default",
    cublas_workspace_mb: int = 0,
    emulation_strategy: str | None = None,
    fixed_point_mantissa_control: str | None = None,
    fixed_point_max_mantissa_bits: int | None = None,
    fixed_point_mantissa_bit_offset: int | None = None,
    streaming_ncsf_cutoff: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute transition (dm1, dm2) on GPU using (segment-walk -> T) then cuBLAS (dm1, cross-Gram).

    Convention matches `asuka.rdm.stream.trans_rdm12_streaming`:
    - dm1[p,q] = <bra| E_{q p} |ket>
    - dm2 uses the same δ-term correction as `make_rdm12`.
    """

    norb = int(drt.norb)
    ncsf = int(drt.ncsf)
    if norb <= 0:
        raise ValueError("invalid norb")
    if ncsf <= 0:
        raise ValueError("invalid ncsf")

    if not has_cuda_ext():
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the CUDA RDM backend") from e

    cbra = np.asarray(ci_bra, dtype=np.float64).ravel()
    cket = np.asarray(ci_ket, dtype=np.float64).ravel()
    if cbra.size != ncsf or cket.size != ncsf:
        raise ValueError("ci_bra/ci_ket have wrong length")

    nops = norb * norb
    block_nops = int(block_nops)
    if block_nops < 1:
        raise ValueError("block_nops must be >= 1")
    block_nops = min(block_nops, nops)

    if workspace is None:
        workspace = make_rdm12_cuda_workspace(
            drt,
            gemm_backend=str(gemm_backend),
            math_mode=str(math_mode),
            cublas_workspace_mb=int(cublas_workspace_mb),
            emulation_strategy=emulation_strategy,
            fixed_point_mantissa_control=fixed_point_mantissa_control,
            fixed_point_max_mantissa_bits=fixed_point_max_mantissa_bits,
            fixed_point_mantissa_bit_offset=fixed_point_mantissa_bit_offset,
        )
    else:
        if int(workspace.norb) != norb or int(workspace.ncsf) != ncsf or int(workspace.nops) != nops:
            raise ValueError("workspace does not match (norb,ncsf)")
        workspace.configure_gram_ws(
            gemm_backend=str(gemm_backend),
            math_mode=str(math_mode),
            cublas_workspace_mb=int(cublas_workspace_mb),
            emulation_strategy=emulation_strategy,
            fixed_point_mantissa_control=fixed_point_mantissa_control,
            fixed_point_max_mantissa_bits=fixed_point_max_mantissa_bits,
            fixed_point_mantissa_bit_offset=fixed_point_mantissa_bit_offset,
        )

    cbra_gpu = cp.asarray(cbra, dtype=cp.float64)
    cket_gpu = cp.asarray(cket, dtype=cp.float64)

    drt_dev = workspace.drt_dev
    state_dev = workspace.state_dev
    p_gpu = workspace.p_gpu
    q_gpu = workspace.q_gpu

    overflow = workspace.overflow
    epq_table = None
    if use_epq_table is not False:
        if use_epq_table is True:
            epq_table = build_epq_action_table_combined_device(
                drt,
                drt_dev,
                state_dev,
                threads=int(build_threads),
                sync=True,
                check_overflow=True,
                use_cache=True,
            )
        else:
            epq_table = getattr(drt, "_epq_action_table_combined_device", None)

    if epq_table is not None:
        use_tiled, tile_csf = _choose_tiling_csf(
            cp,
            ncsf=int(ncsf),
            nops=int(nops),
            full_buffer_factor=2,
            streaming_ncsf_cutoff=streaming_ncsf_cutoff,
        )
        stream = cp.cuda.get_current_stream()
        if use_tiled:
            if workspace.t_tile is None or getattr(workspace.t_tile, "shape", None) != (tile_csf, nops):
                workspace.t_tile = cp.empty((tile_csf, nops), dtype=cp.float64)
            if workspace.t_tile_b is None or getattr(workspace.t_tile_b, "shape", None) != (tile_csf, nops):
                workspace.t_tile_b = cp.empty((tile_csf, nops), dtype=cp.float64)
            if workspace.t_tile_aux is None or getattr(workspace.t_tile_aux, "shape", None) != (tile_csf, nops):
                workspace.t_tile_aux = cp.empty((tile_csf, nops), dtype=cp.float64)
            if workspace.t_tile_aux_b is None or getattr(workspace.t_tile_aux_b, "shape", None) != (tile_csf, nops):
                workspace.t_tile_aux_b = cp.empty((tile_csf, nops), dtype=cp.float64)
            if workspace.dm1_pq_gpu is None or getattr(workspace.dm1_pq_gpu, "shape", None) != (nops,):
                workspace.dm1_pq_gpu = cp.empty((nops,), dtype=cp.float64)
            if workspace.gram0_gpu is None or getattr(workspace.gram0_gpu, "shape", None) != (nops, nops):
                workspace.gram0_gpu = cp.empty((nops, nops), dtype=cp.float64)

            dm1_pq_gpu = workspace.dm1_pq_gpu
            gram0_gpu = workspace.gram0_gpu
            dm1_pq_gpu.fill(0)
            gram0_gpu.fill(0)

            bra_bufs = [workspace.t_tile, workspace.t_tile_b]
            ket_bufs = [workspace.t_tile_aux, workspace.t_tile_aux_b]
            ntiles = (ncsf + tile_csf - 1) // tile_csf

            if ntiles >= 2:
                # Double-buffered pipeline: overlap T-build with GEMM contraction.
                stream_build = cp.cuda.Stream(non_blocking=True)
                stream_contract = cp.cuda.Stream(non_blocking=True)
                init_event = stream.record()
                stream_build.wait_event(init_event)
                stream_contract.wait_event(init_event)
                event_built = [cp.cuda.Event() for _ in range(2)]
                event_contracted = [cp.cuda.Event() for _ in range(2)]

                for tile_idx in range(ntiles):
                    buf_idx = tile_idx % 2
                    k_start = tile_idx * tile_csf
                    k_count = min(tile_csf, ncsf - k_start)
                    t_bra = bra_bufs[buf_idx][:k_count]
                    t_ket = ket_bufs[buf_idx][:k_count]

                    # Wait until previous contraction of THIS buffer is done.
                    if tile_idx >= 2:
                        stream_build.wait_event(event_contracted[buf_idx])

                    # Build both T-tiles on the build stream.
                    cp.cuda.runtime.memsetAsync(
                        int(t_bra.data.ptr), 0,
                        int(t_bra.size) * int(t_bra.itemsize),
                        int(stream_build.ptr),
                    )
                    cp.cuda.runtime.memsetAsync(
                        int(t_ket.data.ptr), 0,
                        int(t_ket.size) * int(t_ket.itemsize),
                        int(stream_build.ptr),
                    )
                    build_w_diag_from_steps_inplace_device(
                        state_dev,
                        j_start=int(k_start),
                        j_count=int(k_count),
                        x=cbra_gpu,
                        w_out=t_bra,
                        threads=256,
                        stream=stream_build,
                        sync=False,
                        relative_w=True,
                    )
                    build_w_diag_from_steps_inplace_device(
                        state_dev,
                        j_start=int(k_start),
                        j_count=int(k_count),
                        x=cket_gpu,
                        w_out=t_ket,
                        threads=256,
                        stream=stream_build,
                        sync=False,
                        relative_w=True,
                    )
                    build_w_from_epq_table_inplace_device(
                        drt,
                        state_dev,
                        epq_table,
                        cbra_gpu,
                        w_out=t_bra,
                        overflow=overflow,
                        threads=int(build_threads),
                        stream=stream_build,
                        sync=False,
                        check_overflow=False,
                        k_start=int(k_start),
                        k_count=int(k_count),
                    )
                    build_w_from_epq_table_inplace_device(
                        drt,
                        state_dev,
                        epq_table,
                        cket_gpu,
                        w_out=t_ket,
                        overflow=overflow,
                        threads=int(build_threads),
                        stream=stream_build,
                        sync=False,
                        check_overflow=False,
                        k_start=int(k_start),
                        k_count=int(k_count),
                    )
                    event_built[buf_idx] = stream_build.record()

                    # Contract on the contract stream (wait for build to finish).
                    stream_contract.wait_event(event_built[buf_idx])
                    cbra_tile = cbra_gpu[k_start : k_start + k_count]
                    rdm_cross_gram_and_dm1_csf_major_inplace_device(
                        workspace.gram_ws,
                        t_bra,
                        t_ket,
                        cbra_tile,
                        dm1_out=dm1_pq_gpu,
                        gram_out=gram0_gpu,
                        stream=stream_contract,
                        sync=False,
                        accumulate=True,
                    )
                    event_contracted[buf_idx] = stream_contract.record()

                # Make the default stream wait for the pipeline to finish.
                stream.wait_event(event_contracted[(ntiles - 1) % 2])
            else:
                # Single tile: no pipeline needed.
                k_count = min(tile_csf, ncsf)
                t_bra = workspace.t_tile[:k_count]
                t_ket = workspace.t_tile_aux[:k_count]

                cp.cuda.runtime.memsetAsync(int(t_bra.data.ptr), 0, int(t_bra.size) * int(t_bra.itemsize), int(stream.ptr))
                cp.cuda.runtime.memsetAsync(int(t_ket.data.ptr), 0, int(t_ket.size) * int(t_ket.itemsize), int(stream.ptr))

                build_w_diag_from_steps_inplace_device(
                    state_dev,
                    j_start=0,
                    j_count=int(k_count),
                    x=cbra_gpu,
                    w_out=t_bra,
                    threads=256,
                    stream=stream,
                    sync=False,
                    relative_w=True,
                )
                build_w_diag_from_steps_inplace_device(
                    state_dev,
                    j_start=0,
                    j_count=int(k_count),
                    x=cket_gpu,
                    w_out=t_ket,
                    threads=256,
                    stream=stream,
                    sync=False,
                    relative_w=True,
                )
                build_w_from_epq_table_inplace_device(
                    drt,
                    state_dev,
                    epq_table,
                    cbra_gpu,
                    w_out=t_bra,
                    overflow=overflow,
                    threads=int(build_threads),
                    stream=stream,
                    sync=False,
                    check_overflow=False,
                    k_start=0,
                    k_count=int(k_count),
                )
                build_w_from_epq_table_inplace_device(
                    drt,
                    state_dev,
                    epq_table,
                    cket_gpu,
                    w_out=t_ket,
                    overflow=overflow,
                    threads=int(build_threads),
                    stream=stream,
                    sync=False,
                    check_overflow=False,
                    k_start=0,
                    k_count=int(k_count),
                )

                cbra_tile = cbra_gpu[:k_count]
                rdm_cross_gram_and_dm1_csf_major_inplace_device(
                    workspace.gram_ws,
                    t_bra,
                    t_ket,
                    cbra_tile,
                    dm1_out=dm1_pq_gpu,
                    gram_out=gram0_gpu,
                    stream=stream,
                    sync=False,
                    accumulate=True,
                )
        else:
            t_bra = cp.empty((ncsf, nops), dtype=cp.float64)
            t_ket = cp.empty((ncsf, nops), dtype=cp.float64)
            cp.cuda.runtime.memsetAsync(int(t_bra.data.ptr), 0, int(t_bra.size) * int(t_bra.itemsize), int(stream.ptr))
            cp.cuda.runtime.memsetAsync(int(t_ket.data.ptr), 0, int(t_ket.size) * int(t_ket.itemsize), int(stream.ptr))

            build_w_diag_from_steps_inplace_device(
                state_dev,
                j_start=0,
                j_count=int(ncsf),
                x=cbra_gpu,
                w_out=t_bra,
                threads=256,
                stream=stream,
                sync=False,
            )
            build_w_diag_from_steps_inplace_device(
                state_dev,
                j_start=0,
                j_count=int(ncsf),
                x=cket_gpu,
                w_out=t_ket,
                threads=256,
                stream=stream,
                sync=False,
            )
            build_w_from_epq_table_inplace_device(
                drt,
                state_dev,
                epq_table,
                cbra_gpu,
                w_out=t_bra,
                overflow=overflow,
                threads=int(build_threads),
                stream=stream,
                sync=False,
                check_overflow=False,
            )
            build_w_from_epq_table_inplace_device(
                drt,
                state_dev,
                epq_table,
                cket_gpu,
                w_out=t_ket,
                overflow=overflow,
                threads=int(build_threads),
                stream=stream,
                sync=False,
                check_overflow=False,
            )

            dm1_pq_gpu = cp.empty((nops,), dtype=cp.float64)
            gram0_gpu = cp.empty((nops, nops), dtype=cp.float64)
            dm1_pq_gpu, gram0_gpu = rdm_cross_gram_and_dm1_csf_major_inplace_device(
                workspace.gram_ws,
                t_bra,
                t_ket,
                cbra_gpu,
                dm1_out=dm1_pq_gpu,
                gram_out=gram0_gpu,
            )
    else:
        t_bra = cp.empty((nops, ncsf), dtype=cp.float64)
        t_ket = cp.empty((nops, ncsf), dtype=cp.float64)
        for i0 in range(0, nops, block_nops):
            i1 = min(nops, i0 + block_nops)
            tb_view = t_bra[i0:i1]
            tk_view = t_ket[i0:i1]

            tb_view, overflow = build_t_block_epq_atomic_inplace_device(
                drt,
                drt_dev,
                state_dev,
                cbra_gpu,
                p_gpu[i0:i1],
                q_gpu[i0:i1],
                out=tb_view,
                overflow=overflow,
                threads=int(build_threads),
                zero_out=True,
                sync=True,
                check_overflow=True,
            )
            tk_view, overflow = build_t_block_epq_atomic_inplace_device(
                drt,
                drt_dev,
                state_dev,
                cket_gpu,
                p_gpu[i0:i1],
                q_gpu[i0:i1],
                out=tk_view,
                overflow=overflow,
                threads=int(build_threads),
                zero_out=True,
                sync=True,
                check_overflow=True,
            )

        dm1_pq_gpu = cp.empty((nops,), dtype=cp.float64)
        gram0_gpu = cp.empty((nops, nops), dtype=cp.float64)
        dm1_pq_gpu, gram0_gpu = rdm_cross_gram_and_dm1_inplace_device(
            workspace.gram_ws,
            t_bra,
            t_ket,
            cbra_gpu,
            dm1_out=dm1_pq_gpu,
            gram_out=gram0_gpu,
        )

    dm1_pq = cp.asnumpy(dm1_pq_gpu)
    gram0 = cp.asnumpy(gram0_gpu).T.copy()

    dm1 = dm1_pq.reshape(norb, norb).T.copy()

    swap = np.arange(nops, dtype=np.int32).reshape(norb, norb).T.ravel()
    gram = gram0[swap]

    dm2 = gram.reshape(norb, norb, norb, norb)
    for p in range(norb):
        for q in range(norb):
            dm2[p, q, q, :] -= dm1[:, p]

    return np.asarray(dm1, dtype=np.float64), np.asarray(dm2, dtype=np.float64)


def trans_rdm1_all_cuda(
    drt: DRT,
    ci_list: list[np.ndarray],
    *,
    build_threads: int = 256,
    workspace: RDM12CudaWorkspace | None = None,
    use_epq_table: bool | None = None,
    streaming_ncsf_cutoff: int | None = None,
) -> np.ndarray:
    """Compute transition dm1 for all bra/ket pairs on GPU.

    Returns
    -------
    dm1_trans:
        Array with shape (nref, nref, norb, norb) in the same convention as
        `asuka.rdm.stream.trans_rdm1_all_streaming`:
          dm1_trans[bra,ket,p,q] = <bra|E_{q p}|ket>
    """

    norb = int(drt.norb)
    ncsf = int(drt.ncsf)
    if norb <= 0:
        raise ValueError("invalid norb")
    if ncsf <= 0:
        raise ValueError("invalid ncsf")
    if not ci_list:
        raise ValueError("ci_list must be non-empty")

    if not has_cuda_ext():
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for trans_rdm1_all_cuda") from e

    nref = int(len(ci_list))
    c_host = [np.asarray(ci, dtype=np.float64).ravel() for ci in ci_list]
    if any(int(c.size) != ncsf for c in c_host):
        raise ValueError("ci_list contains vectors with wrong length")

    nops = norb * norb
    if workspace is None:
        workspace = make_rdm12_cuda_workspace(drt)
    else:
        if int(workspace.norb) != norb or int(workspace.ncsf) != ncsf or int(workspace.nops) != nops:
            raise ValueError("workspace does not match (norb,ncsf)")

    drt_dev = workspace.drt_dev
    state_dev = workspace.state_dev
    overflow = workspace.overflow

    epq_table = None
    if use_epq_table is not False:
        if use_epq_table is True:
            epq_table = build_epq_action_table_combined_device(
                drt,
                drt_dev,
                state_dev,
                threads=int(build_threads),
                sync=True,
                check_overflow=True,
                use_cache=True,
            )
        else:
            epq_table = getattr(drt, "_epq_action_table_combined_device", None)
            if epq_table is None:
                epq_table = build_epq_action_table_combined_device(
                    drt,
                    drt_dev,
                    state_dev,
                    threads=int(build_threads),
                    sync=True,
                    check_overflow=True,
                    use_cache=True,
                )

    if epq_table is None:
        raise RuntimeError("trans_rdm1_all_cuda requires use_epq_table=True")

    c_mat = np.stack(c_host, axis=1)  # (ncsf, nref)
    c_gpu = cp.asarray(c_mat, dtype=cp.float64)
    c_gpu = cp.ascontiguousarray(c_gpu)
    cT_gpu = cp.ascontiguousarray(c_gpu.T)  # (nref, ncsf)

    dm1_trans = np.empty((nref, nref, norb, norb), dtype=np.float64)

    use_tiled, tile_csf = _choose_tiling_csf(
        cp,
        ncsf=int(ncsf),
        nops=int(nops),
        full_buffer_factor=1,
        streaming_ncsf_cutoff=streaming_ncsf_cutoff,
    )
    stream = cp.cuda.get_current_stream()
    if use_tiled:
        if workspace.t_tile is None or getattr(workspace.t_tile, "shape", None) != (tile_csf, nops):
            workspace.t_tile = cp.empty((tile_csf, nops), dtype=cp.float64)
        vals_op = cp.empty((nref, nops), dtype=cp.float64)
        vals_tmp = cp.empty((nref, nops), dtype=cp.float64)

        for ket in range(nref):
            vals_op.fill(0)
            cket_gpu = cp.ascontiguousarray(c_gpu[:, ket])

            for k_start in range(0, ncsf, tile_csf):
                k_count = min(tile_csf, ncsf - k_start)
                t_tile = workspace.t_tile[:k_count]
                cp.cuda.runtime.memsetAsync(
                    int(t_tile.data.ptr),
                    0,
                    int(t_tile.size) * int(t_tile.itemsize),
                    int(stream.ptr),
                )

                build_w_diag_from_steps_inplace_device(
                    state_dev,
                    j_start=int(k_start),
                    j_count=int(k_count),
                    x=cket_gpu,
                    w_out=t_tile,
                    threads=256,
                    stream=stream,
                    sync=False,
                    relative_w=True,
                )
                build_w_from_epq_table_inplace_device(
                    drt,
                    state_dev,
                    epq_table,
                    cket_gpu,
                    w_out=t_tile,
                    overflow=overflow,
                    threads=int(build_threads),
                    stream=stream,
                    sync=False,
                    check_overflow=False,
                    k_start=int(k_start),
                    k_count=int(k_count),
                )

                cT_tile = cT_gpu[:, k_start : k_start + k_count]
                try:
                    cp.matmul(cT_tile, t_tile, out=vals_tmp)
                except Exception:
                    vals_tmp[...] = cp.matmul(cT_tile, t_tile)
                vals_op += vals_tmp

            vals_op_h = np.asarray(cp.asnumpy(vals_op), dtype=np.float64, order="C")
            dm1_trans[:, ket] = vals_op_h.reshape(nref, norb, norb).transpose(0, 2, 1)
    else:
        t_csf = cp.empty((ncsf, nops), dtype=cp.float64)
        for ket in range(nref):
            cp.cuda.runtime.memsetAsync(int(t_csf.data.ptr), 0, int(t_csf.size) * int(t_csf.itemsize), int(stream.ptr))

            cket_gpu = cp.ascontiguousarray(c_gpu[:, ket])
            build_w_diag_from_steps_inplace_device(
                state_dev,
                j_start=0,
                j_count=int(ncsf),
                x=cket_gpu,
                w_out=t_csf,
                threads=256,
                stream=stream,
                sync=False,
            )
            build_w_from_epq_table_inplace_device(
                drt,
                state_dev,
                epq_table,
                cket_gpu,
                w_out=t_csf,
                overflow=overflow,
                threads=int(build_threads),
                stream=stream,
                sync=False,
                check_overflow=False,
            )

            vals_op = cT_gpu @ t_csf  # (nref, nops), vals_op[bra,pq] = <bra|E_pq|ket>
            vals_op_h = np.asarray(cp.asnumpy(vals_op), dtype=np.float64, order="C")
            dm1_trans[:, ket] = vals_op_h.reshape(nref, norb, norb).transpose(0, 2, 1)

    return np.asarray(dm1_trans, dtype=np.float64, order="C")
