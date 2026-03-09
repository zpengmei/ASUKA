from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskList:
    task_spAB: np.ndarray  # int32, shape (ntasks,)
    task_spCD: np.ndarray  # int32, shape (ntasks,)
    task_class_id: np.ndarray | None = None  # int32, shape (ntasks,)

    def __post_init__(self) -> None:
        if self.task_spAB.dtype != np.int32 or self.task_spCD.dtype != np.int32:
            raise TypeError("task_spAB/task_spCD must be int32")
        if self.task_spAB.shape != self.task_spCD.shape or self.task_spAB.ndim != 1:
            raise ValueError("task_spAB/task_spCD must be 1D arrays with identical shape")
        if self.task_class_id is not None:
            if self.task_class_id.dtype != np.int32:
                raise TypeError("task_class_id must be int32")
            if self.task_class_id.shape != self.task_spAB.shape:
                raise ValueError("task_class_id must have shape (ntasks,)")

    @property
    def ntasks(self) -> int:
        return int(self.task_spAB.shape[0])


def build_tasks_screened(Q: np.ndarray, eps: float) -> TaskList:
    """O(nSP^2) screened canonical task list over shell-pairs.

    Returns unique canonical quartets with spCD <= spAB.
    """

    Q = np.asarray(Q, dtype=np.float64).ravel()
    nsp = int(Q.shape[0])
    task_ab: list[int] = []
    task_cd: list[int] = []
    for ab in range(nsp):
        q_ab = float(Q[ab])
        for cd in range(ab + 1):  # canonical: cd <= ab
            if q_ab * float(Q[cd]) >= eps:
                task_ab.append(ab)
                task_cd.append(cd)
    return TaskList(task_spAB=np.asarray(task_ab, dtype=np.int32), task_spCD=np.asarray(task_cd, dtype=np.int32))


def build_tasks_screened_sorted_q(Q: np.ndarray, eps: float) -> TaskList:
    """Vectorized screened canonical task generation.

    Produces the same set as ``build_tasks_screened`` (canonical: spCD ≤ spAB)
    but avoids Python-level loops entirely.

    Algorithm
    ---------
    1. Sort shell pairs by Q descending → perm, Qs.
    2. For each rank-i, find jmax[i] = number of rank-j ≤ i whose
       Qs[j] ≥ eps/Qs[i] via a vectorized np.searchsorted on -Qs.
    3. Build i-array and j-array with np.repeat + offset arithmetic.
    4. Map back to original shell-pair indices via perm and canonicalize.
    """

    Q = np.asarray(Q, dtype=np.float64).ravel()
    nsp = int(Q.shape[0])
    if nsp == 0:
        return TaskList(task_spAB=np.empty((0,), dtype=np.int32), task_spCD=np.empty((0,), dtype=np.int32))

    perm = np.argsort(-Q, kind="stable")  # descending; shape (nsp,)
    Qs = Q[perm]                          # sorted Q values

    # Only consider ranks with Qs[i] > 0 (Q = 0 contributes nothing)
    n_valid = int(np.searchsorted(-Qs, 0.0, side="left"))  # Qs[n_valid-1] > 0 >= Qs[n_valid]
    if n_valid == 0:
        return TaskList(task_spAB=np.empty((0,), dtype=np.int32), task_spCD=np.empty((0,), dtype=np.int32))

    Qs_v = Qs[:n_valid]  # view: all positive Q values in descending order
    neg_Qs_v = -Qs_v      # ascending, for searchsorted

    # jmax[i] = number of rank-j (j=0..n_valid-1) with Qs[j] >= eps/Qs[i].
    # Since Qs is descending, Qs[j] >= thr iff j < searchsorted(-Qs, -thr, 'right').
    # Batch compute: thr[i] = eps / Qs[i]
    thrs = float(eps) / np.maximum(Qs_v, 1e-300)  # (n_valid,)
    jmax_uncapped = np.searchsorted(neg_Qs_v, -thrs, side="right").astype(np.int64)  # (n_valid,)
    # Enforce canonical: j <= i  (in rank-space, i=0..n_valid-1 is the row index)
    i_range = np.arange(n_valid, dtype=np.int64) + 1  # max allowed = i+1
    jmax = np.minimum(jmax_uncapped, i_range)          # (n_valid,)

    # Build pair arrays fully with numpy (no Python loops)
    total = int(jmax.sum())
    if total == 0:
        return TaskList(task_spAB=np.empty((0,), dtype=np.int32), task_spCD=np.empty((0,), dtype=np.int32))

    # i_arr[k] = rank i for the k-th pair; repeat i jmax[i] times
    i_arr = np.repeat(np.arange(n_valid, dtype=np.int64), jmax)  # (total,)

    # j_arr[k] = rank j for the k-th pair: within each group i, j runs 0..jmax[i]-1
    group_offsets = np.empty(n_valid + 1, dtype=np.int64)
    group_offsets[0] = 0
    np.cumsum(jmax, out=group_offsets[1:])
    j_arr = np.arange(total, dtype=np.int64) - group_offsets[i_arr]  # (total,)

    # Map rank indices → original shell-pair indices
    perm32 = perm[:n_valid].astype(np.int32)
    ab_arr = perm32[i_arr]  # (total,) int32
    cd_arr = perm32[j_arr]  # (total,) int32

    # Canonicalize: ensure spCD <= spAB
    swap = cd_arr > ab_arr
    if np.any(swap):
        tmp = ab_arr[swap].copy()
        ab_arr[swap] = cd_arr[swap]
        cd_arr[swap] = tmp

    return TaskList(task_spAB=ab_arr, task_spCD=cd_arr)


def eri_class_id(la: int, lb: int, lc: int, ld: int) -> np.int32:
    """Pack (la,lb,lc,ld) into a single int32 class id.

    Layout (little-endian bytes):
      class_id = la | (lb<<8) | (lc<<16) | (ld<<24)
    """
    for name, l in (("la", la), ("lb", lb), ("lc", lc), ("ld", ld)):
        if l < 0 or l > 255:
            raise ValueError(f"{name} must be in [0,255], got {l}")
    return np.int32((la & 0xFF) | ((lb & 0xFF) << 8) | ((lc & 0xFF) << 16) | ((ld & 0xFF) << 24))


def decode_eri_class_id(class_id: int) -> tuple[int, int, int, int]:
    x = int(class_id) & 0xFFFFFFFF
    return (x & 0xFF, (x >> 8) & 0xFF, (x >> 16) & 0xFF, (x >> 24) & 0xFF)


def compute_task_class_id(task_spAB: np.ndarray, task_spCD: np.ndarray, shell_pairs, shell_l: np.ndarray) -> np.ndarray:
    """Compute per-task (la,lb,lc,ld) class ids.

    Parameters
    - task_spAB/task_spCD: int32 arrays of shell-pair indices (canonical spCD <= spAB)
    - shell_pairs: ShellPairs-like with int32 arrays `sp_A` and `sp_B`
    - shell_l: int32 array of per-shell angular momentum
    """
    task_spAB = np.asarray(task_spAB, dtype=np.int32).ravel()
    task_spCD = np.asarray(task_spCD, dtype=np.int32).ravel()
    if task_spAB.shape != task_spCD.shape:
        raise ValueError("task_spAB and task_spCD must have the same shape")
    shell_l = np.asarray(shell_l, dtype=np.int32).ravel()

    sp_A = np.asarray(shell_pairs.sp_A, dtype=np.int32).ravel()
    sp_B = np.asarray(shell_pairs.sp_B, dtype=np.int32).ravel()

    A = sp_A[task_spAB]
    B = sp_B[task_spAB]
    C = sp_A[task_spCD]
    D = sp_B[task_spCD]

    la = shell_l[A].astype(np.int32)
    lb = shell_l[B].astype(np.int32)
    lc = shell_l[C].astype(np.int32)
    ld = shell_l[D].astype(np.int32)

    return (la & 0xFF) | ((lb & 0xFF) << 8) | ((lc & 0xFF) << 16) | ((ld & 0xFF) << 24)


def with_task_class_id(tasks: TaskList, shell_pairs, shell_l: np.ndarray) -> TaskList:
    """Return a copy of TaskList with `task_class_id` attached."""
    class_id = compute_task_class_id(tasks.task_spAB, tasks.task_spCD, shell_pairs, shell_l)
    return TaskList(task_spAB=tasks.task_spAB, task_spCD=tasks.task_spCD, task_class_id=class_id.astype(np.int32))


def group_tasks_by_class(task_class_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stable-group tasks by `task_class_id`.

    Returns
    - perm: int32 permutation indices (tasks[perm] is grouped by class id)
    - class_ids: int32 unique class ids in group order
    - offsets: int32 array (nclass+1,), offsets into the permuted task list
    """
    task_class_id = np.asarray(task_class_id, dtype=np.int32).ravel()
    nt = int(task_class_id.shape[0])
    if nt == 0:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.asarray([0], dtype=np.int32),
        )

    perm64 = np.argsort(task_class_id, kind="stable")
    perm = np.asarray(perm64, dtype=np.int32)
    cid_sorted = task_class_id[perm]

    changes = np.nonzero(cid_sorted[1:] != cid_sorted[:-1])[0] + 1
    offsets = np.concatenate(([0], changes, [nt])).astype(np.int32)
    class_ids = cid_sorted[offsets[:-1]]
    return perm, class_ids.astype(np.int32), offsets


def group_tasks_by_spab(task_spAB: np.ndarray, nsp: int) -> tuple[np.ndarray, np.ndarray]:
    """Stable-group tasks by `task_spAB` and build an `ab_offsets` CSR-style index.

    Returns
    - perm: int32 permutation indices (tasks[perm] is grouped by spAB)
    - ab_offsets: int32 array (nsp+1,), offsets into the permuted task list for each spAB
    """

    task_spAB = np.asarray(task_spAB, dtype=np.int32).ravel()
    nt = int(task_spAB.shape[0])
    if nsp < 0:
        raise ValueError("nsp must be >= 0")
    if nt == 0:
        return np.empty((0,), dtype=np.int32), np.zeros((nsp + 1,), dtype=np.int32)
    if np.any(task_spAB < 0) or (int(task_spAB.max()) >= nsp):
        raise ValueError("task_spAB entries must be in [0,nsp)")

    perm64 = np.argsort(task_spAB, kind="stable")
    perm = np.asarray(perm64, dtype=np.int32)
    sp_sorted = task_spAB[perm]
    counts = np.bincount(sp_sorted, minlength=nsp).astype(np.int64, copy=False)
    ab_offsets64 = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    return perm, ab_offsets64.astype(np.int32, copy=False)


__all__ = [
    "TaskList",
    "build_tasks_screened",
    "build_tasks_screened_sorted_q",
    "compute_task_class_id",
    "decode_eri_class_id",
    "eri_class_id",
    "group_tasks_by_spab",
    "group_tasks_by_class",
    "with_task_class_id",
]
