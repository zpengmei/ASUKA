import numpy as np
import pytest


@pytest.mark.cuda
def test_hf_df_jk_cuda_ext_k_block_from_qp_d_matches_numpy():
    cp = pytest.importorskip("cupy")

    try:
        from asuka import _hf_df_jk_cuda_ext as ext
    except Exception as e:
        pytest.skip(f"HF DF-JK CUDA extension not available ({type(e).__name__}: {e})")
    if not hasattr(ext, "DFJKWorkspace"):
        pytest.skip("DFJKWorkspace not available in extension")

    ws = ext.DFJKWorkspace()
    if not hasattr(ws, "k_block_from_qp_d"):
        pytest.skip("k_block_from_qp_d not built into extension")

    if int(cp.cuda.runtime.getDeviceCount()) <= 0:
        pytest.skip("no CUDA device")

    rng = np.random.default_rng(0)
    nao = 19
    naux = 23
    row0 = 4
    row_count = 6
    col0 = 2
    col_count = 9
    q_block = 7
    col_block = 5

    B = rng.standard_normal((nao, nao, naux), dtype=np.float64)
    B = 0.5 * (B + B.transpose(1, 0, 2))
    D = rng.standard_normal((nao, nao), dtype=np.float64)
    D = 0.5 * (D + D.T)

    # Reference dense-D K: K = sum_Q B_Q @ D @ B_Q^T
    K_ref = np.zeros((nao, nao), dtype=np.float64)
    for q in range(naux):
        Bq = B[:, :, q]
        K_ref += Bq @ D @ Bq.T

    from asuka.integrals.df_packed_s2 import pack_B_to_Qp

    B_Qp_np = pack_B_to_Qp(B, layout="mnQ", nao=int(nao))
    B_Qp = cp.asarray(B_Qp_np, dtype=cp.float64)
    D_dev = cp.asarray(D, dtype=cp.float64)

    out = cp.empty((row_count, col_count), dtype=cp.float64)
    ws.k_block_from_qp_d(
        B_Qp,
        D_dev,
        out,
        int(nao),
        int(row0),
        int(row_count),
        int(col0),
        int(col_count),
        int(q_block),
        int(col_block),
        stream=int(cp.cuda.get_current_stream().ptr),
        math_mode=-1,
        sync=True,
    )

    got = cp.asnumpy(out)
    ref = K_ref[row0 : row0 + row_count, col0 : col0 + col_count]
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-12)

