import numpy as np


def test_thc_J_factored_matches_dense():
    from asuka.hf.thc_jk import thc_J, thc_J_factored

    rs = np.random.default_rng(0)
    nao = 7
    npt = 9
    r = 3

    X = rs.standard_normal((npt, nao)).astype(np.float64)
    Y = rs.standard_normal((npt, nao)).astype(np.float64)
    Z = (Y @ Y.T).astype(np.float64)

    U = rs.standard_normal((nao, r)).astype(np.float64)
    V = rs.standard_normal((nao, r)).astype(np.float64)
    D = U @ V.T

    J_dense = thc_J(D, X, Z)
    J_yfirst = thc_J(D, X, None, Y=Y)
    J_fact = thc_J_factored(U, V, X, None, Y=Y)

    assert float(np.max(np.abs(J_dense - J_yfirst))) < 1e-12
    max_abs = float(np.max(np.abs(J_dense - J_fact)))
    assert max_abs < 1e-10


def test_thc_K_blocked_factored_matches_dense():
    from asuka.hf.thc_jk import thc_K_blocked, thc_K_blocked_factored

    rs = np.random.default_rng(1)
    nao = 6
    npt = 10
    r = 4

    X = rs.standard_normal((npt, nao)).astype(np.float64)
    Y = rs.standard_normal((npt, nao)).astype(np.float64)
    Z = (Y @ Y.T).astype(np.float64)

    U = rs.standard_normal((nao, r)).astype(np.float64)
    V = rs.standard_normal((nao, r)).astype(np.float64)
    D = U @ V.T

    q_block = 4
    K_dense = thc_K_blocked(D, X, Z, q_block=q_block)
    K_yfirst = thc_K_blocked(D, X, None, q_block=q_block, Y=Y)
    K_fact = thc_K_blocked_factored(U, V, X, None, q_block=q_block, Y=Y)

    assert float(np.max(np.abs(K_dense - K_yfirst))) < 1e-12
    max_abs = float(np.max(np.abs(K_dense - K_fact)))
    assert max_abs < 1e-10


def test_thc_JK_factored_matches_dense():
    from asuka.hf.thc_jk import THCJKWork, thc_JK, thc_JK_factored

    rs = np.random.default_rng(2)
    nao = 7
    npt = 11
    r = 5

    X = rs.standard_normal((npt, nao)).astype(np.float64)
    Y = rs.standard_normal((npt, nao)).astype(np.float64)
    Z = (Y @ Y.T).astype(np.float64)

    U = rs.standard_normal((nao, r)).astype(np.float64)
    V = rs.standard_normal((nao, r)).astype(np.float64)
    D = U @ V.T

    work = THCJKWork(q_block=4)
    J0, K0 = thc_JK(D, X, Z, work=work)
    J0y, K0y = thc_JK(D, X, None, work=work, Y=Y)
    J1, K1 = thc_JK_factored(U, V, X, None, work=work, Y=Y)

    assert float(np.max(np.abs(J0 - J0y))) < 1e-12
    assert float(np.max(np.abs(K0 - K0y))) < 1e-12
    assert float(np.max(np.abs(J0 - J1))) < 1e-10
    assert float(np.max(np.abs(K0 - K1))) < 1e-10


def test_thc_point_gauge_balance_is_exact_in_fp64():
    from asuka.hf.thc_jk import THCJKWork, thc_JK
    from asuka.hf.thc_tc import balance_thc_xy

    rs = np.random.default_rng(3)
    nao = 6
    npt = 8
    naux = 5

    X = rs.standard_normal((npt, nao)).astype(np.float64)
    Y = rs.standard_normal((npt, naux)).astype(np.float64)

    D = rs.standard_normal((nao, nao)).astype(np.float64)
    D = 0.5 * (D + D.T)

    work = THCJKWork(q_block=3)
    J0, K0 = thc_JK(D, X, None, work=work, Y=Y)

    Xb, Yb, _s = balance_thc_xy(X, Y, eps=1e-30, s_min=1e-4, s_max=1e4)
    J1, K1 = thc_JK(D, Xb, None, work=work, Y=Yb)

    assert float(np.max(np.abs(J0 - J1))) < 1e-10
    assert float(np.max(np.abs(K0 - K1))) < 1e-10
