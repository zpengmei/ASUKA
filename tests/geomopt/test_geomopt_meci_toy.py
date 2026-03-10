from __future__ import annotations

import numpy as np

from asuka.geomopt.meci import MECISettings, optimize_meci_cartesian


def test_meci_toy_model_converges():
    # 1 atom -> 3 Cartesian DoF. Interpret coords as (x, y, z) in Bohr.
    k = 1.7

    def multiroot_eval(coords_bohr: np.ndarray):
        coords = np.asarray(coords_bohr, dtype=np.float64).reshape((1, 3))
        x, _y, z = coords[0].tolist()

        # Two smooth "roots" with a degeneracy seam at x=0.
        e0 = 0.5 * k * z * z + x
        e1 = 0.5 * k * z * z - x

        g0 = np.asarray([[1.0, 0.0, k * z]], dtype=np.float64)
        g1 = np.asarray([[-1.0, 0.0, k * z]], dtype=np.float64)

        e_roots = np.asarray([e0, e1], dtype=np.float64)
        grads = np.stack([g0, g1], axis=0)
        return e_roots, grads, None

    def hvec(coords_bohr: np.ndarray, ctx, bra: int, ket: int) -> np.ndarray:
        _ = coords_bohr
        _ = ctx
        _ = bra
        _ = ket
        # Constant h-vector to define a 2D branching plane (x,y). MECI energy depends
        # only on z, so removing motion along h should not affect convergence.
        return np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64)

    coords0 = np.asarray([[0.40, 0.30, -0.25]], dtype=np.float64)
    st = MECISettings(
        max_steps=60,
        penalty_w=20.0,
        gap_tol=1e-8,
        gmax_tol=1e-8,
        grms_tol=1e-8,
        step_max_bohr=0.25,
        verbose=0,
    )

    res = optimize_meci_cartesian(multiroot_eval, coords0, roots=(0, 1), hvec=hvec, settings=st)

    assert res.converged, res.message
    assert abs(res.gap_final) <= 1e-6
    assert abs(float(res.coords_final_bohr[0, 0])) <= 1e-6  # x -> 0 (degeneracy)
    assert abs(float(res.coords_final_bohr[0, 2])) <= 1e-6  # z -> 0 (min Eavg along seam)
    assert abs(float(res.coords_final_bohr[0, 1]) - float(coords0[0, 1])) <= 1e-12  # y unchanged

