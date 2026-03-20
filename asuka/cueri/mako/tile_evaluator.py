"""Mako tile evaluator — dispatches ERI tile evaluation to Mako GEMM kernels.

Selects the appropriate kernel backend based on angular momentum:
  - GEMM kernel (Phase 6):  d-shell classes and below (L ≤ 8)
  - Tiled GEMM (Phase 11):  f/g-shell classes (L > 8)
  - TF32 variant (Phase 10): reduced precision for early SCF iterations

Falls back to the generic scalar kernel for any class not yet optimized.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_AM_LABEL = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}


def _ncart(l: int) -> int:
    return (l + 1) * (l + 2) // 2


def mako_evaluate_tiles(
    ext: Any,
    task_spAB,      # CuPy int32 (ntasks,)
    task_spCD,      # CuPy int32 (ntasks,)
    la: int,
    lb: int,
    lc: int,
    ld: int,
    dsp: Any,       # DeviceShellPairs
    dbasis: Any,    # DeviceBasisSS
    pair_tables: Any,  # DevicePairTables
    *,
    threads: int = 128,
    stream_ptr: int = 0,
    quant_mode: str = "fp64",
    use_fused: bool = False,
    # Fused J/K arguments (Phase 9)
    shell_ao_start: Any = None,
    nao: int = 0,
    D_flat: Any = None,
    J_flat: Any = None,
    K_flat: Any = None,
):
    """Evaluate ERI tiles using the best available Mako kernel.

    Returns
    -------
    tiles : CuPy ndarray or None
        Shape ``(ntasks, ncomp)`` FP64 (or FP32 for TF32 mode).
        Returns *None* when ``use_fused=True`` (tiles consumed in-kernel).
    """
    import cupy as cp  # noqa: PLC0415

    ntasks = int(task_spAB.shape[0])
    if ntasks <= 0:
        ncomp = _ncart(la) * _ncart(lb) * _ncart(lc) * _ncart(ld)
        return cp.empty((0, ncomp), dtype=cp.float64)

    L = la + lb + lc + ld
    max_l = max(la, lb, lc, ld)

    # Phase 9: fused path — no tile materialization
    if use_fused and shell_ao_start is not None and D_flat is not None:
        ext.mako_fused_jk_fp64_device(
            task_spAB, task_spCD, la, lb, lc, ld,
            dsp.sp_A, dsp.sp_B, dsp.sp_pair_start, dsp.sp_npair,
            dbasis.shell_cx, dbasis.shell_cy, dbasis.shell_cz,
            pair_tables.pair_eta, pair_tables.pair_Px,
            pair_tables.pair_Py, pair_tables.pair_Pz,
            pair_tables.pair_cK,
            shell_ao_start, nao, D_flat, J_flat, K_flat,
            threads, stream_ptr,
        )
        return None

    ncomp = _ncart(la) * _ncart(lb) * _ncart(lc) * _ncart(ld)

    # Phase 10b: TF32 tensor core path (FP64 I/O, TF32 precision GEMMs)
    if quant_mode == "tf32tc" and hasattr(ext, "mako_gemm_eri_tf32tc_device"):
        out = cp.empty(ntasks * ncomp, dtype=cp.float64)
        ext.mako_gemm_eri_tf32tc_device(
            task_spAB, task_spCD, la, lb, lc, ld,
            dsp.sp_A, dsp.sp_B, dsp.sp_pair_start, dsp.sp_npair,
            dbasis.shell_cx, dbasis.shell_cy, dbasis.shell_cz,
            pair_tables.pair_eta, pair_tables.pair_Px,
            pair_tables.pair_Py, pair_tables.pair_Pz,
            pair_tables.pair_cK,
            out, threads, stream_ptr,
        )
        return out.reshape(ntasks, ncomp)

    # Phase 10: TF32 path for early SCF iterations
    if quant_mode == "tf32" and hasattr(ext, "mako_gemm_eri_tf32_device"):
        out = cp.empty(ntasks * ncomp, dtype=cp.float32)
        ext.mako_gemm_eri_tf32_device(
            task_spAB, task_spCD, la, lb, lc, ld,
            dsp.sp_A, dsp.sp_B, dsp.sp_pair_start, dsp.sp_npair,
            dbasis.shell_cx, dbasis.shell_cy, dbasis.shell_cz,
            pair_tables.pair_eta, pair_tables.pair_Px,
            pair_tables.pair_Py, pair_tables.pair_Pz,
            pair_tables.pair_cK,
            out, threads, stream_ptr,
        )
        return out.reshape(ntasks, ncomp)

    out = cp.empty(ntasks * ncomp, dtype=cp.float64)

    # Phase 11: tiled GEMM — only when regular GEMM's shared memory exceeds
    # the hardware limit (~96 KB on Ampere/Ada).  The regular GEMM is faster
    # when it fits, so only fall back to tiled for the largest classes (e.g. ffff).
    _smem = ext.mako_gemm_smem_bytes(la, lb, lc, ld) if hasattr(ext, "mako_gemm_smem_bytes") else 0
    if _smem > 96 * 1024 and hasattr(ext, "mako_gemm_tiled_eri_fp64_device"):
        ext.mako_gemm_tiled_eri_fp64_device(
            task_spAB, task_spCD, la, lb, lc, ld,
            dsp.sp_A, dsp.sp_B, dsp.sp_pair_start, dsp.sp_npair,
            dbasis.shell_cx, dbasis.shell_cy, dbasis.shell_cz,
            pair_tables.pair_eta, pair_tables.pair_Px,
            pair_tables.pair_Py, pair_tables.pair_Pz,
            pair_tables.pair_cK,
            out, threads, stream_ptr,
        )
        return out.reshape(ntasks, ncomp)

    # Phase 6: GEMM kernel
    if hasattr(ext, "mako_gemm_eri_fp64_device"):
        ext.mako_gemm_eri_fp64_device(
            task_spAB, task_spCD, la, lb, lc, ld,
            dsp.sp_A, dsp.sp_B, dsp.sp_pair_start, dsp.sp_npair,
            dbasis.shell_cx, dbasis.shell_cy, dbasis.shell_cz,
            pair_tables.pair_eta, pair_tables.pair_Px,
            pair_tables.pair_Py, pair_tables.pair_Pz,
            pair_tables.pair_cK,
            out, threads, stream_ptr,
        )
        return cp.ascontiguousarray(out.reshape(ntasks, ncomp))

    # Fallback: generic scalar kernel
    ext.mako_generic_eri_fp64_device(
        task_spAB, task_spCD, la, lb, lc, ld,
        dsp.sp_A, dsp.sp_B, dsp.sp_pair_start, dsp.sp_npair,
        dbasis.shell_cx, dbasis.shell_cy, dbasis.shell_cz,
        pair_tables.pair_eta, pair_tables.pair_Px,
        pair_tables.pair_Py, pair_tables.pair_Pz,
        pair_tables.pair_cK,
        out, threads, stream_ptr,
    )
    return out.reshape(ntasks, ncomp)
