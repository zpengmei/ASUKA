"""Fused-hop path: single CUDA kernel replaces W-build + GEMM + apply.

This is an alternative to the blocked-EPQ tiling loop in the controller.
It is only available for small active spaces (norb ≤ 20) where a single
DFS-based kernel can compute the entire matvec tile in one launch.

This module is CUDA-specific and NOT part of the base protocol.  New
backends (ROCm) should start with the controller's blocked-EPQ loop and
add fused paths later when equivalent kernels are available.
"""

from __future__ import annotations

from typing import Any

from asuka.cuguga.matvec.protocol import GugaExecutorProtocol


def has_fused_hop() -> bool:
    """Return True if the CUDA extension exposes the fused-hop kernel."""
    from asuka.kernels import guga as guga_kernels

    ext = guga_kernels.load_ext()
    return ext is not None and hasattr(ext, "fused_hop_device")


def is_fused_eligible(executor: GugaExecutorProtocol) -> bool:
    """Return True if the executor's DRT is eligible for fused hop."""
    return has_fused_hop() and executor.norb <= 20
