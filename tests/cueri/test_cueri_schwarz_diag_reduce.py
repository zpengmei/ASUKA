from __future__ import annotations

import numpy as np
import pytest

from asuka.cueri.screening import _diag_max_sqrt_from_square_tiles


@pytest.mark.cuda
@pytest.mark.parametrize("ntasks,n", [(0, 4), (1, 3), (4, 9), (7, 17)])
def test_diag_max_sqrt_kernel_matches_reference(ntasks: int, n: int):
    cp = pytest.importorskip("cupy")
    if int(cp.cuda.runtime.getDeviceCount()) <= 0:
        pytest.skip("no CUDA device")

    rng = np.random.default_rng(20260309 + int(ntasks) * 10 + int(n))
    tile_np = rng.standard_normal((ntasks, n, n), dtype=np.float64)
    tile = cp.asarray(tile_np, dtype=cp.float64)

    got = _diag_max_sqrt_from_square_tiles(tile)
    diag = cp.diagonal(tile, axis1=1, axis2=2)
    ref = cp.sqrt(cp.maximum(cp.max(diag, axis=1), 0.0))

    np.testing.assert_allclose(cp.asnumpy(got), cp.asnumpy(ref), rtol=1e-12, atol=1e-12)

