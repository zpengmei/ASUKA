from __future__ import annotations

"""Native-kernel registry and reporting helpers.

This package provides a *developer-facing* catalog of optional CUDA/C++ kernels
exposed through ASUKA's extension modules (cuERI, HF DF-JK, orbitals, cuGUGA).

Goals:
- centralize extension imports + symbol lookups
- provide a quick "what kernels exist in this build?" report
- make it easier to map Python call sites to kernel symbols in docs

This is not a stable end-user API; production drivers should continue to use
their normal high-level entry points.
"""

from .report import kernel_report, print_kernel_report

__all__ = [
    "kernel_report",
    "print_kernel_report",
]

