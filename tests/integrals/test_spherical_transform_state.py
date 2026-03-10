from types import SimpleNamespace

import numpy as np
import pytest

from asuka.frontend.scf import _apply_sph_transform
from asuka.integrals.cart2sph import AOSphericalTransform, coerce_sph_map
from asuka.integrals.int1e_cart import Int1eResult


def test_coerce_sph_map_accepts_dataclass_and_legacy_tuple():
    T = np.eye(3, dtype=np.float64)
    state = AOSphericalTransform(T_c2s=T, nao_cart=3, nao_sph=3)
    out_state = coerce_sph_map(state)
    assert isinstance(out_state, AOSphericalTransform)
    np.testing.assert_allclose(out_state.T_c2s, T)
    assert int(out_state.nao_cart) == 3
    assert int(out_state.nao_sph) == 3

    out_tuple = coerce_sph_map((T, 3, 3))
    assert isinstance(out_tuple, AOSphericalTransform)
    np.testing.assert_allclose(out_tuple.T_c2s, T)
    assert int(out_tuple.nao_cart) == 3
    assert int(out_tuple.nao_sph) == 3


def test_apply_sph_transform_rejects_l_gt_5():
    mol = SimpleNamespace(cart=False)
    int1e = Int1eResult(S=np.eye(1), T=np.eye(1), V=np.eye(1))
    ao_basis = SimpleNamespace(
        shell_l=np.asarray([6], dtype=np.int32),
        shell_ao_start=np.asarray([0], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="l<=5"):
        _apply_sph_transform(mol, int1e, B=None, ao_basis=ao_basis)
