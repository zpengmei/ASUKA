from __future__ import annotations

import numpy as np

from asuka.density import GridRequest, collect_grid


def _toy_coords() -> np.ndarray:
    # Coordinates in Bohr (arbitrary for these unit tests).
    return np.asarray(
        [
            [0.0, 0.0, -0.70],
            [0.0, 0.0, 0.70],
        ],
        dtype=np.float64,
    )


def test_becke_pruning_reduces_or_preserves_point_count():
    coords = _toy_coords()
    atom_Z = np.asarray([1, 1], dtype=np.int32)

    base = collect_grid(
        coords,
        request=GridRequest(
            kind="becke",
            backend="cpu",
            radial_n=10,
            angular_n=302,
            angular_prune=False,
            atom_Z=atom_Z,
        ),
    )
    pruned = collect_grid(
        coords,
        request=GridRequest(
            kind="becke",
            backend="cpu",
            radial_n=10,
            angular_n=302,
            angular_prune=True,
            atom_Z=atom_Z,
        ),
    )

    assert int(pruned.weights.size) <= int(base.weights.size)
    assert int(pruned.weights.size) > 0
    assert pruned.point_atom is not None
    assert pruned.point_radial_index is not None
    assert pruned.point_angular_n is not None

