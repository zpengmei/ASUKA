from __future__ import annotations

import numpy as np

from asuka.density import GridRequest, collect_grid, iter_grid
from asuka.density.grids import make_becke_grid


def _toy_coords() -> np.ndarray:
    # Coordinates in Bohr (arbitrary for these unit tests).
    return np.asarray(
        [
            [0.0, 0.0, -0.70],
            [0.0, 0.0, 0.70],
        ],
        dtype=np.float64,
    )


def test_collect_grid_becke_cpu_matches_legacy_materialization():
    coords = _toy_coords()
    pts_ref, w_ref = make_becke_grid(coords, radial_n=6, angular_n=38, becke_n=3, prune_tol=1e-16)

    batch = collect_grid(
        coords,
        request=GridRequest(
            kind="becke",
            backend="cpu",
            radial_n=6,
            angular_n=38,
            becke_n=3,
            block_size=100,
            angular_prune=False,
        ),
    )

    np.testing.assert_allclose(batch.points, pts_ref)
    np.testing.assert_allclose(batch.weights, w_ref)
    assert batch.point_atom is not None
    assert int(batch.point_atom.shape[0]) == int(batch.weights.shape[0])


def test_iter_grid_becke_cpu_emits_batches_with_metadata():
    coords = _toy_coords()
    batches = list(
        iter_grid(
            coords,
            request=GridRequest(
                kind="becke",
                backend="cpu",
                radial_n=4,
                angular_n=14,
                block_size=10,
                angular_prune=False,
            ),
        )
    )

    assert len(batches) > 0
    assert all(b.point_atom is not None for b in batches)
    assert all(b.point_radial_index is not None for b in batches)
    assert all(b.point_angular_n is not None for b in batches)
    assert all(int(b.points.shape[0]) == int(b.weights.shape[0]) for b in batches)

