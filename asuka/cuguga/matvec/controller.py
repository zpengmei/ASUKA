"""Backend-neutral matvec controller.

Owns the tiling loop and path selection.  Delegates all device work to an
executor that satisfies :class:`GugaExecutorProtocol`.
"""

from __future__ import annotations

import time
from typing import Any

from asuka.cuguga.matvec.protocol import GugaExecutorProtocol


class GugaMatvecController:
    """Backend-neutral GUGA H*x controller.

    The controller knows about:
      - DRT dimensions (norb, ncsf, nops)
      - include_diagonal_rs flag
      - tiling / blocking strategy

    It does NOT know about:
      - CuPy / CUDA / ROCm APIs
      - Device buffer layout
      - GEMM workspaces
      - Cache policies (CSR, EPQ, CUDA graph, host pinned, ...)

    Those belong to the executor.
    """

    def __init__(
        self,
        executor: GugaExecutorProtocol,
        *,
        include_diagonal_rs: bool = True,
    ) -> None:
        self.executor = executor
        self.norb = executor.norb
        self.ncsf = executor.ncsf
        self.nops = executor.nops
        self.include_diagonal_rs = bool(include_diagonal_rs)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def hop(
        self,
        x: Any,
        *,
        y: Any | None = None,
        stream: Any | None = None,
        sync: bool = True,
        check_overflow: bool = True,
        profile: dict[str, float] | None = None,
    ) -> Any:
        """Compute y = H @ x.

        Uses the minimal blocked-EPQ path:
          1. one-body scatter
          2. for each k-block:
               zero W → build W_diag → build W_offdiag → contract W@ERI → apply g
        """
        ex = self.executor

        # Fail fast: EPQ table is required for the two-body path.
        if not ex.has_epq_table():
            raise RuntimeError(
                "GugaMatvecController requires an EPQ table. "
                "Build the executor with use_epq_table=True."
            )

        if stream is None:
            stream = ex.get_current_stream()

        if y is None:
            y = ex.alloc_y(like=x)

        t_total0 = time.perf_counter() if profile is not None else None

        # --- 1. One-body ---
        t0 = time.perf_counter() if profile is not None else None
        ex.one_body_scatter(x, y, stream=stream)
        if profile is not None and t0 is not None:
            ex.sync_stream(stream)
            profile["one_body_s"] = profile.get("one_body_s", 0.0) + (time.perf_counter() - t0)

        # --- 2. Two-body (blocked EPQ) ---
        epq_table_t = ex.build_epq_transpose()
        block_size = ex.block_size()
        w_buf = ex.alloc_w_block()
        g_buf = ex.alloc_g_block()

        for k0 in range(0, self.ncsf, block_size):
            k_count = min(block_size, self.ncsf - k0)

            # 2a. Zero W
            ex.memzero(w_buf, stream=stream)

            # 2b. Diagonal W (r==s)
            if self.include_diagonal_rs:
                t0 = time.perf_counter() if profile is not None else None
                ex.build_w_diag(x, k0, k_count, w_buf, stream=stream)
                if profile is not None and t0 is not None:
                    ex.sync_stream(stream)
                    profile["diag_w_build_s"] = profile.get("diag_w_build_s", 0.0) + (
                        time.perf_counter() - t0
                    )

            # 2c. Off-diagonal W from EPQ transpose
            t0 = time.perf_counter() if profile is not None else None
            ex.build_w_offdiag(
                epq_table_t, x, k0, k_count, w_buf,
                stream=stream, check_overflow=check_overflow,
            )
            if profile is not None and t0 is not None:
                ex.sync_stream(stream)
                profile["offdiag_w_build_s"] = profile.get("offdiag_w_build_s", 0.0) + (
                    time.perf_counter() - t0
                )

            # 2d. Contract: g = 0.5 * W @ ERI
            t0 = time.perf_counter() if profile is not None else None
            ex.contract_w_eri(w_buf, k_count, g_buf, stream=stream)
            if profile is not None and t0 is not None:
                ex.sync_stream(stream)
                profile["blocked_w_gemm_s"] = profile.get("blocked_w_gemm_s", 0.0) + (
                    time.perf_counter() - t0
                )

            # 2e. Apply: y += E_pq^T @ g
            t0 = time.perf_counter() if profile is not None else None
            ex.apply_g(
                epq_table_t, g_buf, k0, k_count, y,
                stream=stream, check_overflow=check_overflow,
            )
            if profile is not None and t0 is not None:
                ex.sync_stream(stream)
                profile["blocked_w_apply_s"] = profile.get("blocked_w_apply_s", 0.0) + (
                    time.perf_counter() - t0
                )

        if sync:
            ex.sync_stream(stream)

        if profile is not None and t_total0 is not None:
            ex.sync_stream(stream)
            profile["total_s"] = profile.get("total_s", 0.0) + (time.perf_counter() - t_total0)

        return y
