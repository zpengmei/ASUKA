import numpy as np

from asuka.hf.thc_jk import thc_J, thc_K_blocked


def _sym(A):
    return 0.5 * (A + A.T)


def test_thc_k_blocked_matches_explicit_small():
    rng = np.random.default_rng(0)
    nao = 7
    npt = 11

    X = rng.normal(size=(npt, nao))
    D = rng.normal(size=(nao, nao))
    D = _sym(D)

    Z = rng.normal(size=(npt, npt))
    Z = _sym(Z)

    # Explicit exchange contraction:
    #   M = X D X^T
    #   K = X^T ( (Z ⊙ M) X )
    M = X @ D @ X.T
    K_exp = X.T @ ((Z * M) @ X)
    K_exp = _sym(K_exp)

    K_blk = thc_K_blocked(D, X, Z, q_block=4)

    assert K_blk.shape == (nao, nao)
    np.testing.assert_allclose(K_blk, K_exp, rtol=1e-12, atol=1e-12)


def test_thc_j_symmetry_small():
    rng = np.random.default_rng(1)
    nao = 6
    npt = 9

    X = rng.normal(size=(npt, nao))
    D = _sym(rng.normal(size=(nao, nao)))
    Z = _sym(rng.normal(size=(npt, npt)))

    J = thc_J(D, X, Z)
    assert J.shape == (nao, nao)
    np.testing.assert_allclose(J, J.T, rtol=0.0, atol=1e-12)

