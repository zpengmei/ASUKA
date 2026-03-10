import numpy as np

from asuka.hf import df_jk


def _k_dense_ref_mega_gemm_mnQ(B: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Reference dense-D DF-K matching the legacy mega-GEMM contraction order."""

    nao = int(D.shape[0])
    naux = int(B.shape[2])
    B2d = B.reshape(nao, nao * naux)
    BQD = np.tensordot(B, D, axes=([1], [0]))  # (nao, naux, nao)
    M1 = np.ascontiguousarray(BQD.transpose(0, 2, 1)).reshape(nao, nao * naux)
    return M1 @ B2d.T


def test_df_k_from_cocc_matches_dense_core_density():
    rng = np.random.default_rng(0)

    nao = 11
    naux = 23
    ncore = 4

    B = rng.standard_normal((nao, nao, naux), dtype=np.float64)
    C_core = rng.standard_normal((nao, ncore), dtype=np.float64)
    D_core = 2.0 * (C_core @ C_core.T)

    K_ref = _k_dense_ref_mega_gemm_mnQ(B, D_core)
    occ_core = np.full((ncore,), 2.0, dtype=np.float64)
    K_cocc = df_jk.df_K_from_BmnQ_Cocc(B, C_core, occ_core, q_block=7)

    assert np.allclose(K_cocc, K_ref, rtol=1e-12, atol=1e-12)


def test_df_k_from_cocc_matches_dense_active_density_via_eigh():
    rng = np.random.default_rng(1)

    nao = 9
    naux = 19
    ncas = 5

    B = rng.standard_normal((nao, nao, naux), dtype=np.float64)
    C_act = rng.standard_normal((nao, ncas), dtype=np.float64)

    # Build a PSD active 1-RDM with eigenvalues in [0, 2].
    Q, _ = np.linalg.qr(rng.standard_normal((ncas, ncas)))
    w = rng.random((ncas,), dtype=np.float64) * 2.0
    dm1_act = Q @ np.diag(w) @ Q.T

    D_act = C_act @ dm1_act @ C_act.T
    K_ref = _k_dense_ref_mega_gemm_mnQ(B, D_act)

    w_h, U_h = np.linalg.eigh(0.5 * (dm1_act + dm1_act.T))
    w_h = np.clip(w_h, 0.0, None)
    C_no = C_act @ U_h
    K_cocc = df_jk.df_K_from_BmnQ_Cocc(B, C_no, w_h, q_block=8)

    assert np.allclose(K_cocc, K_ref, rtol=1e-12, atol=1e-12)

