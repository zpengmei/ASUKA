from __future__ import annotations

import os
import warnings
from typing import Any


def configure_offdiag_gemm_workspace(
    *,
    ws: Any,
    gemm_backend: str,
    offdiag_enable_fp64_emulation: bool,
    offdiag_emulation_strategy: str | None,
) -> None:
    """Configure offdiag GEMM backend/math-mode policy on a workspace."""
    if bool(offdiag_enable_fp64_emulation):
        ws.set_gemm_backend("gemmex_emulated_fixedpoint")
        ws.set_cublas_math_mode("fp64_emulated_fixedpoint")
        if offdiag_emulation_strategy:
            strategy = str(offdiag_emulation_strategy).strip().lower()
            if strategy == "eager":
                allow = str(os.getenv("CUGUGA_ALLOW_CUBLAS_EAGER", "")).strip().lower()
                if allow not in ("1", "true", "yes"):
                    raise RuntimeError(
                        "offdiag_emulation_strategy='eager' is experimental and may crash for some shapes/GPUs; "
                        "set CUGUGA_ALLOW_CUBLAS_EAGER=1 to enable anyway"
                    )
            ws.set_cublas_emulation_strategy(str(strategy))
        return

    backend_name = str(gemm_backend)
    ws.set_gemm_backend(backend_name)
    if backend_name not in ("gemmex_tf32", "cublaslt_tf32"):
        ws.set_cublas_math_mode("default")


def autoset_offdiag_cublas_workspace_with_backoff(
    *,
    ws: Any,
    nops: int,
    nrows_eff: int,
    cap_mb: int,
    recommend_fn: Any,
) -> dict[str, int]:
    """Auto-size offdiag cuBLAS workspace with OOM backoff and warning parity."""
    cap_bytes = None
    if int(cap_mb) > 0:
        cap_bytes = int(cap_mb) * 1024 * 1024

    ws_info = ws.cublas_emulation_info()
    rec = recommend_fn(
        ws_info=ws_info,
        gemm_shapes=[(int(nops), int(nrows_eff), int(nops))],
        batch_count=1,
        is_complex=False,
        cap_bytes=cap_bytes,
    )
    requested_ws = int(rec)
    applied_ws = int(requested_ws)
    if applied_ws > 0:
        # Best-effort backoff for constrained GPUs.
        step_bytes = 64 * 1024 * 1024
        while True:
            try:
                ws.set_cublas_workspace_bytes(int(applied_ws))
                break
            except Exception as e:
                msg = str(e).lower()
                is_oom = ("out of memory" in msg) or ("memory" in msg) or ("alloc" in msg)
                if not is_oom:
                    raise
                next_ws = (int(applied_ws) // 2 // int(step_bytes)) * int(step_bytes)
                if next_ws <= 0:
                    applied_ws = 0
                    ws.set_cublas_workspace_bytes(0)
                    break
                applied_ws = int(next_ws)
    else:
        ws.set_cublas_workspace_bytes(0)

    if int(applied_ws) < int(requested_ws):
        warnings.warn(
            f"CUDA offdiag emulation workspace reduced from {requested_ws // (1024 * 1024)} MiB "
            f"to {applied_ws // (1024 * 1024)} MiB due to device memory limits",
            RuntimeWarning,
        )

    return {
        "requested_ws": int(requested_ws),
        "applied_ws": int(applied_ws),
        "workspace_bytes": int(ws.cublas_workspace_bytes()),
    }
