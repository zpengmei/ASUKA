import numpy as np
import pytest


@pytest.mark.cuda
def test_cuda_spawn_event_compaction_helpers():
    cp = pytest.importorskip("cupy")

    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CUDA runtime unavailable ({type(e).__name__}: {e})")
    if ndev <= 0:
        pytest.skip("no CUDA device")

    from asuka.qmc.cuda_backend import _compact_spawn_events_i32, _compact_spawn_events_u64

    evt_idx = cp.asarray([3, -1, 7, 8, -1, 10], dtype=cp.int32)
    evt_val = cp.asarray([1.0, 2.0, 0.0, -0.5, 0.0, 3.0], dtype=cp.float64)
    idx_c, val_c, n_keep = _compact_spawn_events_i32(evt_idx, evt_val)
    assert n_keep == 3
    np.testing.assert_array_equal(cp.asnumpy(idx_c), np.asarray([3, 8, 10], dtype=np.int32))
    np.testing.assert_allclose(cp.asnumpy(val_c), np.asarray([1.0, -0.5, 3.0], dtype=np.float64), rtol=0, atol=0)

    invalid = np.uint64(0xFFFFFFFFFFFFFFFF)
    evt_key = cp.asarray(
        [np.uint64(2), invalid, np.uint64(9), np.uint64(11), invalid, np.uint64(13)],
        dtype=cp.uint64,
    )
    evt_val_u64 = cp.asarray([0.5, 1.5, 0.0, -2.0, 0.0, 4.0], dtype=cp.float64)
    key_c, val_c_u64, n_keep_u64 = _compact_spawn_events_u64(evt_key, evt_val_u64)
    assert n_keep_u64 == 3
    np.testing.assert_array_equal(cp.asnumpy(key_c), np.asarray([2, 11, 13], dtype=np.uint64))
    np.testing.assert_allclose(
        cp.asnumpy(val_c_u64),
        np.asarray([0.5, -2.0, 4.0], dtype=np.float64),
        rtol=0,
        atol=0,
    )
