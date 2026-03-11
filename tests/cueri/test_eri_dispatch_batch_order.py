from __future__ import annotations

import numpy as np

from asuka.cueri.eri_dispatch import plan_kernel_batches_spd
from asuka.cueri.shell_pairs import ShellPairs
from asuka.cueri.tasks import TaskList, eri_class_id


def test_plan_kernel_batches_spd_sorts_each_group_by_pair_work_stably():
    shell_l = np.asarray([1, 1, 1, 0, 0], dtype=np.int32)
    shell_pairs = ShellPairs(
        sp_A=np.asarray([0, 0, 1, 2, 0], dtype=np.int32),
        sp_B=np.asarray([1, 3, 3, 3, 4], dtype=np.int32),
        sp_npair=np.asarray([4, 6, 2, 2, 10], dtype=np.int32),
        sp_pair_start=np.asarray([0, 4, 10, 12, 14, 24], dtype=np.int32),
    )
    tasks = TaskList(
        task_spAB=np.asarray([0, 0, 0, 0], dtype=np.int32),
        task_spCD=np.asarray([1, 2, 3, 4], dtype=np.int32),
        task_class_id=np.full((4,), int(eri_class_id(1, 1, 1, 0)), dtype=np.int32),
    )

    batches = plan_kernel_batches_spd(tasks, shell_pairs=shell_pairs, shell_l=shell_l)

    assert len(batches) == 1
    batch = batches[0]
    np.testing.assert_array_equal(batch.task_idx, np.asarray([1, 2, 0, 3], dtype=np.int32))
    np.testing.assert_array_equal(batch.kernel_tasks.task_spAB, np.asarray([0, 0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(batch.kernel_tasks.task_spCD, np.asarray([2, 3, 1, 4], dtype=np.int32))


def test_plan_kernel_batches_spd_keeps_transpose_scatter_semantics_when_sorting():
    shell_l = np.asarray([2, 1, 1, 0, 0], dtype=np.int32)
    shell_pairs = ShellPairs(
        sp_A=np.asarray([0, 0, 1, 2], dtype=np.int32),
        sp_B=np.asarray([3, 4, 3, 4], dtype=np.int32),
        sp_npair=np.asarray([9, 15, 2, 5], dtype=np.int32),
        sp_pair_start=np.asarray([0, 9, 24, 26, 31], dtype=np.int32),
    )
    tasks = TaskList(
        task_spAB=np.asarray([0, 1], dtype=np.int32),
        task_spCD=np.asarray([2, 3], dtype=np.int32),
        task_class_id=np.full((2,), int(eri_class_id(2, 0, 1, 0)), dtype=np.int32),
    )

    batches = plan_kernel_batches_spd(tasks, shell_pairs=shell_pairs, shell_l=shell_l)

    assert len(batches) == 1
    batch = batches[0]
    assert batch.transpose is True
    assert int(batch.kernel_class_id) == int(eri_class_id(1, 0, 2, 0))
    np.testing.assert_array_equal(batch.task_idx, np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(batch.kernel_tasks.task_spAB, np.asarray([2, 3], dtype=np.int32))
    np.testing.assert_array_equal(batch.kernel_tasks.task_spCD, np.asarray([0, 1], dtype=np.int32))
