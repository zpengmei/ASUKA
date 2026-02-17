from __future__ import annotations

import numpy as np
import pytest

from asuka.semiempirical import am1_energy, am1_energy_gradient
from asuka.semiempirical.params import ANGSTROM_TO_BOHR

_CASES = [
    ("h2", ["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)),
    (
        "hcn",
        ["H", "C", "N"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.06], [0.0, 0.0, 2.22]], dtype=float),
    ),
    (
        "ch4",
        ["C", "H", "H", "H", "H"],
        np.array(
            [
                [0.0000, 0.0000, 0.0000],
                [0.6291, 0.6291, 0.6291],
                [0.6291, -0.6291, -0.6291],
                [-0.6291, 0.6291, -0.6291],
                [-0.6291, -0.6291, 0.6291],
            ],
            dtype=float,
        ),
    ),
    (
        "h2o",
        ["O", "H", "H"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]], dtype=float),
    ),
    ("co", ["C", "O"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]], dtype=float)),
    ("n2", ["N", "N"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.10]], dtype=float)),
]


@pytest.mark.parametrize("case_name,symbols,coords_ang", _CASES)
def test_am1_gradient_directional_fd_consistency(case_name, symbols, coords_ang):
    del case_name
    coords_bohr = np.asarray(coords_ang, dtype=float) * ANGSTROM_TO_BOHR
    e0, grad = am1_energy_gradient(
        symbols,
        coords_bohr,
        coords_unit="bohr",
        max_iter=120,
        conv_tol=1e-9,
        fock_mode="ri",
    )
    assert np.isfinite(float(e0))
    assert np.all(np.isfinite(grad))
    assert np.max(np.abs(np.sum(grad, axis=0))) <= 1e-6

    # Use a deterministic random direction projected to remove net translation.
    rng = np.random.default_rng(7 + len(symbols))
    direction = rng.normal(size=coords_bohr.shape)
    direction -= np.mean(direction, axis=0, keepdims=True)
    direction /= np.linalg.norm(direction)

    h = 2e-4
    e_p = am1_energy(
        symbols,
        coords_bohr + h * direction,
        coords_unit="bohr",
        max_iter=120,
        conv_tol=1e-9,
        fock_mode="ri",
    )
    e_m = am1_energy(
        symbols,
        coords_bohr - h * direction,
        coords_unit="bohr",
        max_iter=120,
        conv_tol=1e-9,
        fock_mode="ri",
    )
    fd_proj = (float(e_p) - float(e_m)) / (2.0 * h)
    grad_proj = float(np.sum(np.asarray(grad, dtype=float) * direction))
    assert abs(fd_proj - grad_proj) <= 5e-5

