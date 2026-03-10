import numpy as np

from asuka.hf import df_scf


def _k_mega_gemm_mnQ(B: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Reference implementation matching the legacy GPU mega-GEMM contraction.

    K[m,k] = sum_Q sum_{p,n} B[m,p,Q] * D[p,n] * B[k,n,Q]
    """

    nao = int(D.shape[0])
    naux = int(B.shape[2])
    B2d = B.reshape(nao, nao * naux)
    BQD = np.tensordot(B, D, axes=([1], [0]))  # (nao, naux, nao) with axes (m,Q,n)
    M1 = np.ascontiguousarray(BQD.transpose(0, 2, 1)).reshape(nao, nao * naux)
    return M1 @ B2d.T


def test_df_k_qblocked_matches_mega_gemm():
    rng = np.random.default_rng(0)

    nao = 13
    naux = 29
    B = rng.standard_normal((nao, nao, naux), dtype=np.float64)
    D = rng.standard_normal((nao, nao), dtype=np.float64)

    K_ref = _k_mega_gemm_mnQ(B, D)

    # Exercise non-trivial blocking (not a divisor of naux).
    K_blk = df_scf._df_K_qblocked_mnQ(B, D, q_block=8)
    assert np.allclose(K_blk, K_ref, rtol=1e-12, atol=1e-12)


def test_df_k_qblocked_qblock_ge_naux_is_valid():
    rng = np.random.default_rng(1)

    nao = 7
    naux = 11
    B = rng.standard_normal((nao, nao, naux), dtype=np.float64)
    D = rng.standard_normal((nao, nao), dtype=np.float64)

    K_ref = _k_mega_gemm_mnQ(B, D)
    K_blk = df_scf._df_K_qblocked_mnQ(B, D, q_block=128)
    assert np.allclose(K_blk, K_ref, rtol=1e-12, atol=1e-12)

