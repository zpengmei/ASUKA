"""Typed containers for CASPT2 gradient intermediates.

These are intentionally lightweight (NumPy array holders) so they can be used
as stable return types across SS/MS/XMS gradient driver implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MSGradientIntermediates:
    """Intermediate tensors for MS/XMS-CASPT2 gradient assembly.

    Notes
    -----
    This container is used by tests as a shape/metadata round-trip gate.  It
    does not enforce strict dtype/contiguity; callers should normalise arrays
    upstream when required for performance.
    """

    clag_full: np.ndarray
    olag_total: np.ndarray
    dpt2_1rdm_total: np.ndarray
    dpt2c_total: np.ndarray
    dpt2_2rdm_total: np.ndarray
    olag_pre: np.ndarray
    slag: np.ndarray
    wlag: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)

