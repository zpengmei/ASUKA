import numpy as np
import pytest


@pytest.mark.cuda
def test_df_pack_lf_block_to_qp_device_matches_numpy():
    cp = pytest.importorskip("cupy")

    try:
        from asuka.cueri import _cueri_cuda_ext as ext
    except Exception as e:
        import os
        import sys

        pytest.skip(
            "cuERI CUDA extension not available "
            f"({type(e).__name__}: {e}; cwd={os.getcwd()}; sys.path[0:5]={sys.path[0:5]!r})"
        )
    if not hasattr(ext, "df_pack_lf_block_to_qp_device"):
        pytest.skip("df_pack_lf_block_to_qp_device not built into extension")

    if int(cp.cuda.runtime.getDeviceCount()) <= 0:
        pytest.skip("no CUDA device")

    rs = np.random.RandomState(0)
    nao = 33
    naux = 23
    q0 = 7
    q_count = 5
    ntri = nao * (nao + 1) // 2

    Lf_np = rs.standard_normal((nao, q_count * nao)).astype(np.float64, copy=False)
    out_np = np.zeros((naux, ntri), dtype=np.float64)

    tri_i, tri_j = np.tril_indices(nao)
    for q_local in range(q_count):
        q_abs = q0 + q_local
        out_np[q_abs, :] = Lf_np[tri_i, q_local * nao + tri_j]

    Lf = cp.asarray(Lf_np)
    out = cp.zeros((naux, ntri), dtype=cp.float64)

    threads = 256
    stream_ptr = int(cp.cuda.get_current_stream().ptr)
    ext.df_pack_lf_block_to_qp_device(
        Lf.reshape(-1),
        out.reshape(-1),
        int(naux),
        int(nao),
        int(q0),
        int(q_count),
        int(threads),
        int(stream_ptr),
        True,  # sync
    )

    got = cp.asnumpy(out)
    assert got.shape == out_np.shape
    assert np.array_equal(got, out_np)
