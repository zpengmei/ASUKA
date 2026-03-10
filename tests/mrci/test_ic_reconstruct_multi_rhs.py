from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from asuka.mrci.ic_basis import ICDoubles, ICSingles, OrbitalSpaces
from asuka.mrci.ic_reconstruct import ic_mrcisd_multi_reference_ci_rhs_from_residual


def _empty_ic_singles() -> ICSingles:
    return ICSingles(
        a=np.asarray([], dtype=np.int32),
        r=np.asarray([], dtype=np.int32),
        a_group_offsets=np.asarray([0], dtype=np.int32),
        a_group_order=np.asarray([], dtype=np.int32),
        a_group_keys=np.asarray([], dtype=np.int32),
    )


def _empty_ic_doubles() -> ICDoubles:
    return ICDoubles(
        a=np.asarray([], dtype=np.int32),
        b=np.asarray([], dtype=np.int32),
        r=np.asarray([], dtype=np.int32),
        s=np.asarray([], dtype=np.int32),
        ab_group_offsets=np.asarray([0], dtype=np.int32),
        ab_group_order=np.asarray([], dtype=np.int32),
        ab_group_keys=np.asarray([], dtype=np.int32).reshape((0, 2)),
    )


def test_multi_reference_ci_rhs_keeps_block_specific_reference_weights(monkeypatch):
    import asuka.mrci.ic_reconstruct as mod

    drt = SimpleNamespace(
        ncsf=4,
        nelec=2,
        twos_target=0,
        node_sym=np.asarray([0], dtype=np.int32),
        leaf=0,
    )

    residual = np.asarray([2.0, -1.0, 0.5, -0.25], dtype=np.float64)
    coeff = np.asarray([[1.0, 2.0]], dtype=np.float64)
    spaces = [
        OrbitalSpaces(internal=np.asarray([0], dtype=np.int32), external=np.asarray([], dtype=np.int32)),
        OrbitalSpaces(internal=np.asarray([0], dtype=np.int32), external=np.asarray([], dtype=np.int32)),
    ]
    ic_res = SimpleNamespace(
        c=coeff,
        e=np.asarray([0.5], dtype=np.float64),
        block_slices=[(0, 1), (1, 2)],
        singles=[_empty_ic_singles(), _empty_ic_singles()],
        doubles=[_empty_ic_doubles(), _empty_ic_doubles()],
        spaces=spaces,
        allow_same_internal=True,
    )

    monkeypatch.setattr(
        "asuka.mrci.ic_mrcisd.expand_ic_mrcisd_multi_root",
        lambda _ic_res, *, ci_cas, root: (drt, np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)),
    )
    monkeypatch.setattr(
        "asuka.mrci.ic_mrcisd._build_uncontracted_hop",
        lambda **kwargs: (lambda x: residual + float(ic_res.e[0]) * np.asarray(x, dtype=np.float64)),
    )
    monkeypatch.setattr("asuka.cuguga.oracle._get_epq_action_cache", lambda _drt: None)
    monkeypatch.setattr(mod, "build_drt", lambda **kwargs: drt)
    monkeypatch.setattr(
        mod,
        "embed_cas_ci_into_mrcisd",
        lambda **kwargs: (
            np.asarray([1.0, 0.0], dtype=np.float64),
            np.asarray([1.0, 0.0], dtype=np.float64),
            np.asarray([0, 1], dtype=np.int32),
        ),
    )

    rhs = ic_mrcisd_multi_reference_ci_rhs_from_residual(
        ic_res,
        ci_cas=[
            np.asarray([1.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 1.0], dtype=np.float64),
        ],
        root=0,
        h1e=np.zeros((1, 1), dtype=np.float64),
        eri=np.zeros((1, 1, 1, 1), dtype=np.float64),
    )

    assert len(rhs) == 2
    np.testing.assert_allclose(rhs[0], np.asarray([4.0, -2.0], dtype=np.float64), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(rhs[1], np.asarray([8.0, -4.0], dtype=np.float64), atol=1e-12, rtol=1e-12)
