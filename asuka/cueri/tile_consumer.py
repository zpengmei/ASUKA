from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class TileConsumer(Protocol):
    """Consumer interface for evaluated ERI tiles.

    A tile consumer "digests" evaluated contracted Cartesian tiles into some output
    representation (e.g., packed ERIs, Fock/J/K contributions, intermediates).
    """

    def consume(
        self,
        *,
        task_idx: np.ndarray,
        task_spAB: np.ndarray,
        task_spCD: np.ndarray,
        kernel_class_id: np.int32,
        tiles: Any,
    ) -> None:
        ...

    def finalize(self) -> Any:
        ...


__all__ = ["TileConsumer"]
