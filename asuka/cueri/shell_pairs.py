from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShellPairs:
    """Canonical shell-pair list with A>=B."""

    sp_A: np.ndarray  # int32, shape (nSP,)
    sp_B: np.ndarray  # int32, shape (nSP,)
    sp_npair: np.ndarray  # int32, shape (nSP,) = nprim(A)*nprim(B)
    sp_pair_start: np.ndarray  # int32, shape (nSP+1,)

    def __post_init__(self) -> None:
        for name, arr in (
            ("sp_A", self.sp_A),
            ("sp_B", self.sp_B),
            ("sp_npair", self.sp_npair),
            ("sp_pair_start", self.sp_pair_start),
        ):
            if arr.dtype != np.int32:
                raise TypeError(f"{name} must be int32")
        if self.sp_A.shape != self.sp_B.shape or self.sp_A.ndim != 1:
            raise ValueError("sp_A/sp_B must be 1D arrays with identical shape")
        nsp = int(self.sp_A.shape[0])
        if self.sp_npair.shape != (nsp,):
            raise ValueError("sp_npair must have shape (nSP,)")
        if self.sp_pair_start.shape != (nsp + 1,):
            raise ValueError("sp_pair_start must have shape (nSP+1,)")
        if int(self.sp_pair_start[0]) != 0:
            raise ValueError("sp_pair_start[0] must be 0")
        if np.any(self.sp_pair_start[1:] < self.sp_pair_start[:-1]):
            raise ValueError("sp_pair_start must be non-decreasing")
        if int(self.sp_pair_start[-1]) != int(np.sum(self.sp_npair)):
            raise ValueError("sp_pair_start[-1] must equal sum(sp_npair)")


def build_shell_pairs(basis) -> ShellPairs:
    """Build canonical shell pairs for a packed basis (Step 1/2).

    The input must provide:
    - `shell_cxyz`: float64 array (nShell,3)
    - `shell_nprim`: int32 array (nShell,)
    """

    n_shell = int(basis.shell_cxyz.shape[0])
    sp_A: list[int] = []
    sp_B: list[int] = []
    sp_npair: list[int] = []
    sp_pair_start: list[int] = [0]

    for A in range(n_shell):
        nprim_A = int(basis.shell_nprim[A])
        for B in range(A + 1):  # canonical A>=B
            nprim_B = int(basis.shell_nprim[B])
            sp_A.append(A)
            sp_B.append(B)
            npair = nprim_A * nprim_B
            sp_npair.append(npair)
            sp_pair_start.append(sp_pair_start[-1] + npair)

    return ShellPairs(
        sp_A=np.asarray(sp_A, dtype=np.int32),
        sp_B=np.asarray(sp_B, dtype=np.int32),
        sp_npair=np.asarray(sp_npair, dtype=np.int32),
        sp_pair_start=np.asarray(sp_pair_start, dtype=np.int32),
    )

def build_shell_pairs_l_order(basis) -> ShellPairs:
    """Build shell pairs with l-ordered orientation (for class-specialized kernels).

    Produces the same *set* of unordered shell pairs as `build_shell_pairs` but orients each
    pair such that `shell_l[A] >= shell_l[B]`. This is useful for kernels that assume the
    higher angular momentum shell is the first index of the pair.

    The input must provide:
    - `shell_l`: int32 array (nShell,)
    - `shell_nprim`: int32 array (nShell,)
    """

    shell_l = np.asarray(basis.shell_l, dtype=np.int32).ravel()
    n_shell = int(shell_l.shape[0])

    sp_A: list[int] = []
    sp_B: list[int] = []
    sp_npair: list[int] = []
    sp_pair_start: list[int] = [0]

    for i in range(n_shell):
        nprim_i = int(basis.shell_nprim[i])
        li = int(shell_l[i])
        for j in range(i + 1):  # unique unordered pairs
            nprim_j = int(basis.shell_nprim[j])
            lj = int(shell_l[j])

            A, B = i, j
            nprim_A, nprim_B = nprim_i, nprim_j
            if li < lj:
                A, B = j, i
                nprim_A, nprim_B = nprim_j, nprim_i

            sp_A.append(int(A))
            sp_B.append(int(B))
            npair = int(nprim_A) * int(nprim_B)
            sp_npair.append(npair)
            sp_pair_start.append(sp_pair_start[-1] + npair)

    return ShellPairs(
        sp_A=np.asarray(sp_A, dtype=np.int32),
        sp_B=np.asarray(sp_B, dtype=np.int32),
        sp_npair=np.asarray(sp_npair, dtype=np.int32),
        sp_pair_start=np.asarray(sp_pair_start, dtype=np.int32),
    )


__all__ = ["ShellPairs", "build_shell_pairs", "build_shell_pairs_l_order"]
