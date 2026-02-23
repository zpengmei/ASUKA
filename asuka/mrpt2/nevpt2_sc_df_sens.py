"""Adjoint / sensitivity scaffolding for SC-NEVPT2(DF).

The public API in :mod:`asuka.mrpt2` historically exported an adjoint-style
entry point for an upcoming analytic-gradient implementation.

In the snapshot shipped with this chat session, only the *finite-difference*
gradient driver is implemented (see :func:`asuka.mrpt2.nevpt2_sc_df_grad_from_ref`).

This module is provided so that ``import asuka.mrpt2`` succeeds even when the
adjoint backend is not yet implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NEVPT2SCDFAdjointTotalResult:
    """Placeholder result container for an adjoint SC-NEVPT2(DF) implementation."""

    backend: str
    details: dict[str, Any] | None = None


def nevpt2_sc_total_energy_df_adjoint(*args: Any, **kwargs: Any) -> NEVPT2SCDFAdjointTotalResult:
    """Placeholder for an adjoint (analytic-gradient) SC-NEVPT2(DF) backend.

    The analytic NEVPT2 gradient implementation is not part of this snapshot.
    Use :func:`asuka.mrpt2.nevpt2_sc_df_grad_from_ref` with
    ``grad_backend='fd'`` for finite-difference gradients.
    """

    raise NotImplementedError(
        "Analytic/adjoint SC-NEVPT2(DF) sensitivity is not implemented in this snapshot. "
        "Use finite-difference gradients: nevpt2_sc_df_grad_from_ref(grad_backend='fd')."
    )
