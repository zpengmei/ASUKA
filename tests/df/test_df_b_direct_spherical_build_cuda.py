import numpy as np
import pytest


@pytest.mark.cuda
def test_df_b_direct_spherical_build_matches_cart_transform_cuda():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.zeros((1,), dtype=cp.float64)
        _ = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        pytest.skip("CuPy is present but a CUDA device is unavailable")

    try:
        from asuka.cueri import _cueri_cuda_ext as _ext  # noqa: PLC0415
    except Exception:
        pytest.skip("cuERI CUDA extension is unavailable")
    if not hasattr(_ext, "scatter_df_int3c2e_tiles_cart_to_sph_inplace_device"):
        pytest.skip("cuERI CUDA extension lacks spherical DF scatter support (rebuild extension)")

    from asuka.frontend import Molecule
    from asuka.frontend.df import build_df_bases_cart
    from asuka.integrals.cart2sph import build_cart2sph_matrix, compute_sph_layout_from_cart_basis, transform_df_B_cart_to_sph
    from asuka.integrals.cueri_df import build_df_B_from_cueri_packed_bases

    mol = Molecule.from_atoms(
        atoms=[
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, -1.43233673, 1.10715266)),
            ("H", (0.0, 1.43233673, 1.10715266)),
        ],
        basis="6-31g*",
        cart=False,
    )
    ao_basis, aux_basis, _aux_name = build_df_bases_cart(mol, auxbasis="autoaux", expand_contractions=True)

    B_cart = build_df_B_from_cueri_packed_bases(ao_basis, aux_basis, layout="mnQ", ao_rep="cart")
    assert int(B_cart.shape[0]) == int(B_cart.shape[1])
    nao_cart = int(B_cart.shape[0])

    shell_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    shell_ao_start_cart = np.asarray(ao_basis.shell_ao_start, dtype=np.int32).ravel()
    shell_ao_start_sph, nao_sph = compute_sph_layout_from_cart_basis(ao_basis)
    assert int(nao_sph) < int(nao_cart)
    T = build_cart2sph_matrix(shell_l, shell_ao_start_cart, shell_ao_start_sph, nao_cart, int(nao_sph))

    B_sph_ref = transform_df_B_cart_to_sph(
        B_cart,
        T,
        shell_l=shell_l,
        shell_ao_start_cart=shell_ao_start_cart,
        shell_ao_start_sph=shell_ao_start_sph,
        out_layout="mnQ",
    )
    B_sph = build_df_B_from_cueri_packed_bases(ao_basis, aux_basis, layout="mnQ", ao_rep="sph")

    assert tuple(map(int, B_sph.shape)) == tuple(map(int, B_sph_ref.shape))
    denom = float(cp.linalg.norm(B_sph_ref).item())
    denom = max(1.0, denom)
    rel = float((cp.linalg.norm(B_sph - B_sph_ref) / denom).item())
    assert rel < 1e-8

