"""Executor protocol for GUGA matvec backends.

Any backend (CUDA, ROCm, CPU-reference) implements this protocol.
The controller calls only these methods — never touches device APIs directly.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GugaExecutorProtocol(Protocol):
    """Minimal interface a matvec backend must implement.

    All array arguments are opaque to the controller — the executor decides
    whether they live on GPU, NPU, or host.  The controller only passes them
    through and never inspects their dtype/device.

    ``stream`` is backend-specific (cupy.cuda.Stream, torch_npu stream, etc.).
    The controller treats it as an opaque handle.
    """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def norb(self) -> int: ...

    @property
    def ncsf(self) -> int: ...

    @property
    def nops(self) -> int: ...

    # ------------------------------------------------------------------
    # Buffer lifecycle
    # ------------------------------------------------------------------

    def alloc_y(self, *, like: Any) -> Any:
        """Allocate an output vector y with same shape/dtype/device as *like*."""
        ...

    def memzero(self, buf: Any, *, stream: Any) -> None:
        """Async memset *buf* to zero on *stream*."""
        ...

    # ------------------------------------------------------------------
    # One-body contribution
    # ------------------------------------------------------------------

    def one_body_scatter(
        self,
        x: Any,
        y: Any,
        *,
        stream: Any,
    ) -> None:
        """Compute y = h_eff @ x (one-body), zeroing y first.

        The executor uses its internally stored h_eff.
        """
        ...

    # ------------------------------------------------------------------
    # EPQ table management
    # ------------------------------------------------------------------

    def has_epq_table(self) -> bool:
        """Return True if a pre-built EPQ action table is available."""
        ...

    def build_epq_transpose(self) -> Any:
        """Build and return the destination-major EPQ transpose table."""
        ...

    # ------------------------------------------------------------------
    # Two-body blocked path (minimal trunk)
    # ------------------------------------------------------------------

    def build_w_diag(
        self,
        x: Any,
        k_start: int,
        k_count: int,
        w_out: Any,
        *,
        stream: Any,
    ) -> None:
        """Fill diagonal (r==s) occupation entries of W[k_start:k_start+k_count, :]."""
        ...

    def build_w_offdiag(
        self,
        epq_table_t: Any,
        x: Any,
        k_start: int,
        k_count: int,
        w_out: Any,
        *,
        stream: Any,
        check_overflow: bool = True,
    ) -> None:
        """Fill off-diagonal W entries from EPQ transpose table."""
        ...

    def contract_w_eri(
        self,
        w: Any,
        k_count: int,
        g_out: Any,
        *,
        stream: Any,
    ) -> None:
        """g = 0.5 * W @ ERI  (or DF two-step).  Writes into g_out[:k_count]."""
        ...

    def apply_g(
        self,
        epq_table_t: Any,
        g_block: Any,
        k_start: int,
        k_count: int,
        y: Any,
        *,
        stream: Any,
        check_overflow: bool = True,
    ) -> None:
        """y += E_pq^T g  via destination-major EPQ transpose gather."""
        ...

    # ------------------------------------------------------------------
    # Block-size query
    # ------------------------------------------------------------------

    def block_size(self) -> int:
        """Return the max number of ket rows per block (controls memory pressure)."""
        ...

    def alloc_w_block(self) -> Any:
        """Allocate (or return cached) W buffer of shape (block_size, nops)."""
        ...

    def alloc_g_block(self) -> Any:
        """Allocate (or return cached) g buffer of shape (block_size, nops)."""
        ...

    # ------------------------------------------------------------------
    # Synchronisation
    # ------------------------------------------------------------------

    def sync_stream(self, stream: Any) -> None:
        """Block until all work on *stream* completes."""
        ...

    def get_current_stream(self) -> Any:
        """Return the default execution stream for this backend."""
        ...

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self) -> int:
        """Free device resources.  Return estimated bytes freed."""
        ...
