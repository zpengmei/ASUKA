from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.cuda
def test_task_arrays_device_cached_reuses_uploaded_arrays():
    cp = pytest.importorskip("cupy")
    if int(cp.cuda.runtime.getDeviceCount()) <= 0:
        pytest.skip("no CUDA device")

    from asuka.cueri.gpu import _task_arrays_device_cached
    from asuka.cueri.tasks import TaskList

    tasks = TaskList(
        task_spAB=np.asarray([0, 3, 2, 1], dtype=np.int32),
        task_spCD=np.asarray([0, 1, 1, 0], dtype=np.int32),
    )

    ab0, cd0 = _task_arrays_device_cached(tasks)
    ab1, cd1 = _task_arrays_device_cached(tasks)

    assert isinstance(ab0, cp.ndarray)
    assert isinstance(cd0, cp.ndarray)
    assert ab0.dtype == cp.int32 and cd0.dtype == cp.int32
    assert bool(ab0.flags.c_contiguous) and bool(cd0.flags.c_contiguous)
    assert ab1 is ab0 and cd1 is cd0
    np.testing.assert_array_equal(cp.asnumpy(ab0), np.asarray(tasks.task_spAB, dtype=np.int32))
    np.testing.assert_array_equal(cp.asnumpy(cd0), np.asarray(tasks.task_spCD, dtype=np.int32))


@pytest.mark.cuda
def test_task_arrays_device_cached_passthrough_when_already_device():
    cp = pytest.importorskip("cupy")
    if int(cp.cuda.runtime.getDeviceCount()) <= 0:
        pytest.skip("no CUDA device")

    from asuka.cueri.gpu import _task_arrays_device_cached
    from asuka.cueri.tasks import TaskList

    task_ab = cp.ascontiguousarray(cp.asarray([1, 0, 4], dtype=cp.int32))
    task_cd = cp.ascontiguousarray(cp.asarray([0, 0, 2], dtype=cp.int32))
    tasks = TaskList(task_spAB=task_ab, task_spCD=task_cd)

    ab, cd = _task_arrays_device_cached(tasks)
    assert ab is task_ab
    assert cd is task_cd
