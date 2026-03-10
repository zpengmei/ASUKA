from __future__ import annotations

import numpy as np

from asuka.mrci.generalized_davidson import generalized_davidson
from asuka.mrci.ic_mrcisd import expand_ic_mrcisd_multi_root, ic_mrcisd_kernel_multi


def test_generalized_davidson_multi_solves_lowest_generalized_roots():
    h = np.diag([1.0, 2.0, 4.0, 6.0])
    s = np.diag([1.0, 2.0, 1.0, 1.0])

    res = generalized_davidson(
        lambda x: h @ x,
        lambda x: s @ x,
        [np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0])],
        nroots=2,
        max_cycle=20,
        max_space=8,
        tol=1e-12,
    )

    np.testing.assert_allclose(np.asarray(res.e), np.asarray([1.0, 1.0]), atol=1e-10, rtol=1e-10)
    assert np.asarray(res.converged, dtype=bool).tolist() == [True, True]


def test_expand_ic_mrcisd_multi_root_trivial_single_block():
    ic_res = ic_mrcisd_kernel_multi(
        h1e=np.zeros((1, 1), dtype=np.float64),
        eri=np.zeros((1, 1, 1, 1), dtype=np.float64),
        n_act=1,
        n_virt=0,
        nelec=2,
        twos=0,
        ci_cas=[np.asarray([1.0], dtype=np.float64)],
        nroots=1,
        contraction="fic",
        backend="semi_direct",
        solver="dense",
        max_cycle=5,
        max_space=4,
    )

    drt, ci = expand_ic_mrcisd_multi_root(
        ic_res,
        ci_cas=[np.asarray([1.0], dtype=np.float64)],
        root=0,
    )

    assert int(drt.ncsf) == 1
    np.testing.assert_allclose(np.asarray(ci), np.asarray([1.0]), atol=1e-12, rtol=1e-12)
