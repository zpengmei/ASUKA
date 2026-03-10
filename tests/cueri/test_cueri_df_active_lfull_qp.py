from __future__ import annotations

import numpy as np
import pytest

from asuka.cueri.df import active_Lfull_from_B, active_Lfull_from_B_Qp
from asuka.integrals.df_packed_s2 import pack_B_to_Qp


@pytest.mark.cuda
@pytest.mark.parametrize("aux_block_naux", [1, 5, 64, -3])
def test_active_lfull_from_b_qp_matches_full_path(aux_block_naux: int):
    cp = pytest.importorskip("cupy")
    try:
        from asuka.kernels.cueri import require_ext
    except Exception as e:
        pytest.skip(f"cuERI extension unavailable ({type(e).__name__}: {e})")
    ext = require_ext()
    if not hasattr(ext, "df_fused_qp_l_act_device"):
        pytest.skip("cuERI extension lacks df_fused_qp_l_act_device")
    if int(cp.cuda.runtime.getDeviceCount()) <= 0:
        pytest.skip("no CUDA device")

    rng = np.random.default_rng(20260309 + int(aux_block_naux))
    nao, norb, naux = 13, 5, 29
    B_mnQ_np = rng.standard_normal((nao, nao, naux), dtype=np.float64)
    B_mnQ_np = 0.5 * (B_mnQ_np + B_mnQ_np.transpose(1, 0, 2))
    C_np = rng.standard_normal((nao, norb), dtype=np.float64)

    B_qp_np = pack_B_to_Qp(B_mnQ_np, layout="mnQ", nao=int(nao))

    B_mnQ = cp.asarray(B_mnQ_np, dtype=cp.float64)
    B_qp = cp.asarray(B_qp_np, dtype=cp.float64)
    C_act = cp.asarray(C_np, dtype=cp.float64)

    ref = active_Lfull_from_B(B_mnQ, C_act)
    got = active_Lfull_from_B_Qp(B_qp, C_act, aux_block_naux=int(aux_block_naux))

    np.testing.assert_allclose(cp.asnumpy(got), cp.asnumpy(ref), rtol=1e-10, atol=1e-10)

