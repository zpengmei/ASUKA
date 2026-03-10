import numpy as np

from asuka.hf import df_jk
from asuka.integrals.df_packed_s2 import pack_B_to_Qp


def test_df_k_from_cocc_qp_matches_mnq():
    rng = np.random.default_rng(123)

    nao = 13
    naux = 29
    nocc = 5

    B = rng.standard_normal((nao, nao, naux), dtype=np.float64)
    # DF factors are symmetric in (mu,nu); packing assumes this.
    B = 0.5 * (B + B.transpose(1, 0, 2))

    C_occ = rng.standard_normal((nao, nocc), dtype=np.float64)
    occ_vals = rng.random((nocc,), dtype=np.float64) * 2.0

    K_ref = df_jk.df_K_from_BmnQ_Cocc(B, C_occ, occ_vals, q_block=7)

    B_Qp = pack_B_to_Qp(B, layout="mnQ", nao=int(nao))
    assert B_Qp.ndim == 2
    K_qp = df_jk.df_K_from_BmnQ_Cocc(B_Qp, C_occ, occ_vals, q_block=7)

    assert np.allclose(K_qp, K_ref, rtol=1e-12, atol=1e-12)

