from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

GridKind = Literal["becke", "rdvr", "fdvr", "cube"]
GridBackend = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class GridRequest:
    """Canonical request object for ASUKA numerical grids.

    Notes
    -----
    - Not every field is used by every grid kind.
    - `angular_prune=None` means "use kind default" (rdvr=True, becke=False).
    """

    kind: GridKind = "rdvr"
    backend: GridBackend = "auto"

    # Common numerical controls
    radial_n: int = 50
    angular_n: int = 302
    angular_kind: str = "auto"
    rmax: float = 20.0
    becke_n: int = 3
    block_size: int = 20_000
    prune_tol: float = 1e-16
    threads: int = 256
    stream: Any | None = None

    # R-DVR / pruning controls
    radial_rmax: float | None = None
    ortho_cutoff: float = 1e-10
    angular_prune: bool | None = None
    atom_Z: Any | None = None

    # F-DVR controls
    validate: bool = True
    overlap_max_abs_tol: float = 1e-3
    max_sweeps: int = 50
    tol: float = 1e-12

    # Cube-grid controls
    spacing: float = 0.25
    padding: float = 4.0


@dataclass(slots=True)
class GridBatch:
    """One batch of grid points plus optional provenance arrays.

    Arrays may be NumPy or CuPy depending on the backend.
    """

    points: Any
    weights: Any
    point_atom: Any | None = None
    point_radial_index: Any | None = None
    point_angular_n: Any | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def npt(self) -> int:
        return int(getattr(self.weights, "shape", [0])[0])

