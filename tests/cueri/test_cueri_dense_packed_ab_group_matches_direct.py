from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.cuda
def test_cueri_dense_packed_ab_group_matches_direct():
    cp = pytest.importorskip("cupy")
    if int(cp.cuda.runtime.getDeviceCount()) <= 0:
        pytest.skip("no CUDA device")

    try:
        from asuka.cueri.gpu import has_cuda_ext
    except Exception as e:
        pytest.skip(f"cuERI GPU module unavailable ({type(e).__name__}: {e})")
    if not bool(has_cuda_ext()):
        pytest.skip("cuERI CUDA extension unavailable")

    from asuka.frontend.molecule import Molecule
    from asuka.frontend.one_electron import build_ao_basis_cart
    from asuka.integrals.int1e_cart import nao_cart_from_basis
    from asuka.cueri.dense import build_active_eri_packed_dense_sp_only

    mol = Molecule.from_atoms(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))],
        unit="Bohr",
        basis="sto-3g",
        cart=True,
    )
    ao_basis, _ = build_ao_basis_cart(mol, basis=mol.basis, expand_contractions=True)

    rng = np.random.default_rng(20260309)
    nao = int(nao_cart_from_basis(ao_basis))
    ncas = min(2, nao)
    C = cp.asarray(rng.standard_normal((nao, ncas), dtype=np.float64))

    packed_direct = build_active_eri_packed_dense_sp_only(
        ao_basis,
        C,
        threads=256,
        max_tile_bytes=64 * 1024 * 1024,
        eps_ao=0.0,
        eps_mo=0.0,
        algorithm="direct",
    )
    packed_ab_group = build_active_eri_packed_dense_sp_only(
        ao_basis,
        C,
        threads=256,
        max_tile_bytes=64 * 1024 * 1024,
        eps_ao=0.0,
        eps_mo=0.0,
        algorithm="ab_group",
    )

    np.testing.assert_allclose(
        cp.asnumpy(packed_ab_group),
        cp.asnumpy(packed_direct),
        rtol=1e-10,
        atol=1e-10,
    )
