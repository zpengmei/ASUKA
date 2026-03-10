import numpy as np

from asuka.integrals.df_adjoint import df_whiten_adjoint_Qmn


def test_df_whiten_adjoint_qmn_overwrite_matches_non_overwrite():
    rng = np.random.default_rng(0)

    nao = 6
    naux = 9

    B = rng.standard_normal((nao, nao, naux), dtype=np.float64)
    bar_L = rng.standard_normal((naux, nao, nao), dtype=np.float64)

    # Any invertible lower-triangular L is fine for this algebraic test.
    L = rng.standard_normal((naux, naux), dtype=np.float64)
    L = np.tril(L)
    L[np.diag_indices_from(L)] += 5.0

    bar_X0, bar_Lchol0 = df_whiten_adjoint_Qmn(B, bar_L.copy(order="C"), L, overwrite_bar_L=False)
    bar_X1, bar_Lchol1 = df_whiten_adjoint_Qmn(B, bar_L.copy(order="C"), L, overwrite_bar_L=True)

    assert np.allclose(bar_X0, bar_X1, rtol=1e-12, atol=1e-12)
    assert np.allclose(bar_Lchol0, bar_Lchol1, rtol=1e-12, atol=1e-12)

