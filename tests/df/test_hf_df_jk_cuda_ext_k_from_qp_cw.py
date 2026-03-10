import numpy as np
import pytest

from asuka.hf import df_jk


@pytest.mark.cuda
def test_hf_df_jk_cuda_ext_k_from_qp_cw_matches_numpy():
    cp = pytest.importorskip("cupy")

    try:
        from asuka import _hf_df_jk_cuda_ext as ext
    except Exception as e:
        pytest.skip(f"HF DF-JK CUDA extension not available ({type(e).__name__}: {e})")
    if not hasattr(ext, "DFJKWorkspace"):
        pytest.skip("DFJKWorkspace not available in extension")

    ws = ext.DFJKWorkspace()
    if not hasattr(ws, "k_from_qp_cw"):
        pytest.skip("k_from_qp_cw not built into extension")

    if int(cp.cuda.runtime.getDeviceCount()) <= 0:
        pytest.skip("no CUDA device")

    rng = np.random.default_rng(1)
    nao = 17
    naux = 29
    nocc = 6
    q_block = 7

    B = rng.standard_normal((nao, nao, naux), dtype=np.float64)
    B = 0.5 * (B + B.transpose(1, 0, 2))
    C_occ = rng.standard_normal((nao, nocc), dtype=np.float64)
    occ_vals = rng.random((nocc,), dtype=np.float64) * 2.0

    K_ref = df_jk.df_K_from_BmnQ_Cocc(B, C_occ, occ_vals, q_block=int(q_block))

    from asuka.integrals.df_packed_s2 import pack_B_to_Qp

    B_Qp_np = pack_B_to_Qp(B, layout="mnQ", nao=int(nao))
    B_Qp = cp.asarray(B_Qp_np, dtype=cp.float64)
    Cw = cp.asarray(C_occ * np.sqrt(occ_vals)[None, :], dtype=cp.float64)

    out = cp.empty((nao, nao), dtype=cp.float64)
    ws.k_from_qp_cw(
        B_Qp,
        Cw,
        out,
        q_block=int(q_block),
        stream=int(cp.cuda.get_current_stream().ptr),
        math_mode=-1,
        sync=True,
    )

    got = cp.asnumpy(out)
    assert np.allclose(got, K_ref, rtol=1e-12, atol=1e-12)

