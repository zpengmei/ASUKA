from __future__ import annotations

"""Analytical cost model for cuERI direct-J/K contraction kernels.

The goal is not to predict wall time exactly. Instead, this module computes
algorithm-level operation / traffic counts for the contraction kernels given the
tile shape and shell-pair symmetry pattern used by :mod:`asuka.hf.direct_jk`.

The model is intentionally limited to kernels whose math is explicit in the
source tree:

- staged element-wise direct J/K contraction
- warp-reduce direct J/K contraction
- the corresponding multi-density variants via ``n_dm``

It does **not** attempt to symbolically count the ERI-evaluation recurrence
flops of the generated Rys kernels. For that stage, the most reliable exact
quantity is the number of ERI values materialized and the tile round-trip
traffic between ERI evaluation and contraction. The arithmetic counts below are
exact with respect to the explicit contraction formulas and their symmetry
scalings, but they are not meant to be literal SASS/PTX instruction counts.
"""

from dataclasses import asdict, dataclass, replace
from typing import Iterable

import numpy as np


# Symmetry histogram index encoding.
# bit2 = ab_neq, bit1 = cd_neq, bit0 = bra_ket_swap
_HIST_SIZE = 8


@dataclass(frozen=True)
class DirectJKContractCost:
    """Exact contraction-side operation and traffic counts.

    All counts are totals over the provided task histogram.

    Parameters
    ----------
    contract_mode
        ``"staged"`` for the element-wise dense scatter kernels,
        ``"warp"`` for the warp-reduced contraction kernels.
    tile_roundtrip_bytes
        Global-memory traffic associated with materializing the ERI tile before
        contraction.  For staged kernels this includes the ERI write from the
        evaluation kernel plus the tile reads performed by the contraction
        kernel.  Fused kernels avoid this round-trip.
    output_rmw_bytes_lower_bound
        Lower bound on global traffic from atomics, assuming a read-modify-write
        of one 8-byte value per atomic update.
    """

    contract_mode: str
    n_dm: int
    ntasks: int
    nA: int
    nB: int
    nC: int
    nD: int
    eri_values: int
    tile_loads: int
    tile_bytes_read: int
    tile_roundtrip_bytes: int
    density_loads: int
    density_bytes: int
    inner_fmuls: int
    scale_fmuls: int
    fmuls: int
    reduction_adds: int
    output_atomics: int
    output_payload_bytes: int
    output_rmw_bytes_lower_bound: int
    flops: int
    arithmetic_intensity_payload: float
    arithmetic_intensity_rmw_lower_bound: float

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class DirectJKSymmetryCase:
    ntasks: int
    ab_neq: bool
    cd_neq: bool
    bra_ket_swap: bool


def symmetry_histogram_from_masks(ab_neq, cd_neq, bra_ket_swap) -> np.ndarray:
    """Return a compact 8-bin symmetry histogram.

    Parameters are broadcast-compatible 1D boolean arrays on the host.
    """

    ab = np.asarray(ab_neq, dtype=np.int8).reshape(-1)
    cd = np.asarray(cd_neq, dtype=np.int8).reshape(-1)
    bk = np.asarray(bra_ket_swap, dtype=np.int8).reshape(-1)
    if ab.shape != cd.shape or ab.shape != bk.shape:
        raise ValueError("ab_neq/cd_neq/bra_ket_swap must have identical shape")
    case = (ab << 2) | (cd << 1) | bk
    hist = np.bincount(case, minlength=_HIST_SIZE).astype(np.int64, copy=False)
    if hist.shape != (_HIST_SIZE,):
        hist = np.resize(hist, (_HIST_SIZE,)).astype(np.int64, copy=False)
    return hist


def symmetry_histogram_from_cases(cases: Iterable[DirectJKSymmetryCase]) -> np.ndarray:
    hist = np.zeros((_HIST_SIZE,), dtype=np.int64)
    for case in cases:
        nt = int(case.ntasks)
        if nt <= 0:
            continue
        idx = (
            (int(bool(case.ab_neq)) << 2)
            | (int(bool(case.cd_neq)) << 1)
            | int(bool(case.bra_ket_swap))
        )
        hist[idx] += nt
    return hist


def _case_count(hist: np.ndarray, *, ab_neq: bool, cd_neq: bool, bra_ket_swap: bool) -> int:
    idx = (
        (int(bool(ab_neq)) << 2)
        | (int(bool(cd_neq)) << 1)
        | int(bool(bra_ket_swap))
    )
    return int(np.asarray(hist, dtype=np.int64).reshape(_HIST_SIZE)[idx])


def estimate_direct_jk_contract_cost(
    *,
    histogram: np.ndarray,
    nA: int,
    nB: int,
    nC: int,
    nD: int,
    contract_mode: str,
    want_J: bool,
    want_K: bool,
    n_dm: int = 1,
) -> DirectJKContractCost:
    """Estimate exact contraction-kernel cost from an 8-bin symmetry histogram.

    The histogram bins track whether the shell pairs are off-diagonal on the bra
    side, off-diagonal on the ket side, and whether the shell-pair indices differ
    (triggering the bra-ket swap contribution).
    """

    mode = str(contract_mode).strip().lower().replace("-", "_")
    if mode not in {"staged", "warp"}:
        raise ValueError("contract_mode must be 'staged' or 'warp'")
    if not bool(want_J) and not bool(want_K):
        raise ValueError("at least one of want_J/want_K must be True")

    hist = np.asarray(histogram, dtype=np.int64).reshape(-1)
    if hist.size != _HIST_SIZE:
        raise ValueError("histogram must have 8 bins")

    nA_i = int(nA)
    nB_i = int(nB)
    nC_i = int(nC)
    nD_i = int(nD)
    if min(nA_i, nB_i, nC_i, nD_i) <= 0:
        raise ValueError("nA/nB/nC/nD must be > 0")
    n_dm_i = int(n_dm)
    if n_dm_i <= 0:
        raise ValueError("n_dm must be > 0")

    nAB = nA_i * nB_i
    nCD = nC_i * nD_i
    eri_values = 0
    tile_loads = 0
    density_loads = 0
    inner_fmuls = 0
    scale_fmuls = 0
    fmuls = 0
    reduction_adds = 0
    output_atomics = 0
    ntasks = int(hist.sum(dtype=np.int64))

    for idx, count_raw in enumerate(hist.tolist()):
        count = int(count_raw)
        if count <= 0:
            continue
        ab_neq = bool((idx >> 2) & 0x1)
        cd_neq = bool((idx >> 1) & 0x1)
        bra_ket_swap = bool(idx & 0x1)

        m = int(count) * int(nAB) * int(nCD)
        eri_values += m

        j_terms = (1 + int(bra_ket_swap)) if bool(want_J) else 0
        k_terms = (
            1 + int(cd_neq) + int(ab_neq) + int(ab_neq and cd_neq)
        ) if bool(want_K) else 0

        per_dm_inner_mul = int(m) * int(j_terms + k_terms)
        density_loads += int(n_dm_i) * int(per_dm_inner_mul)
        inner_fmuls += int(n_dm_i) * int(per_dm_inner_mul)

        if mode == "staged":
            # Dense scatter kernel walks each ERI value once and reuses it across
            # all requested density matrices handled inside the kernel.
            tile_loads += int(m)
            scale_fmuls += int(n_dm_i) * int(m) * int(j_terms)
            j_atomic_factor = 0
            if bool(want_J):
                j_atomic_factor = (1 + int(ab_neq)) + int(bra_ket_swap) * (1 + int(cd_neq))
            k_atomic_factor = 0
            if bool(want_K):
                k_primary = 1 + int(cd_neq) + int(ab_neq) + int(ab_neq and cd_neq)
                k_atomic_factor = k_primary * (1 + int(bra_ket_swap))
            output_atomics += int(n_dm_i) * int(m) * int(j_atomic_factor + k_atomic_factor)
        else:
            # Warp contraction re-reads the tile for each reduction family.
            tile_loads += int(n_dm_i) * int(per_dm_inner_mul)

            j_scale_task = 0
            j_atomic_task = 0
            j_reduction_task = 0
            if bool(want_J):
                j_scale_task = nAB
                j_atomic_task = nAB * (1 + int(ab_neq))
                if bra_ket_swap:
                    j_scale_task += nCD
                    j_atomic_task += nCD * (1 + int(cd_neq))
                j_reduction_task = nAB * max(nCD - 1, 0)
                if bra_ket_swap:
                    j_reduction_task += nCD * max(nAB - 1, 0)

            k_atomic_task = 0
            k_reduction_task = 0
            if bool(want_K):
                k_atomic_task = nA_i * nC_i * (1 + int(bra_ket_swap))
                k_reduction_task = nA_i * nC_i * max(nB_i * nD_i - 1, 0)
                if cd_neq:
                    k_atomic_task += nA_i * nD_i * (1 + int(bra_ket_swap))
                    k_reduction_task += nA_i * nD_i * max(nB_i * nC_i - 1, 0)
                if ab_neq:
                    k_atomic_task += nB_i * nC_i * (1 + int(bra_ket_swap))
                    k_reduction_task += nB_i * nC_i * max(nA_i * nD_i - 1, 0)
                if ab_neq and cd_neq:
                    k_atomic_task += nB_i * nD_i * (1 + int(bra_ket_swap))
                    k_reduction_task += nB_i * nD_i * max(nA_i * nC_i - 1, 0)

            scale_fmuls += int(n_dm_i) * int(count) * int(j_scale_task)
            output_atomics += int(n_dm_i) * int(count) * int(j_atomic_task + k_atomic_task)
            reduction_adds += int(n_dm_i) * int(count) * int(j_reduction_task + k_reduction_task)

    fmuls = int(inner_fmuls) + int(scale_fmuls)

    tile_bytes_read = 8 * int(tile_loads)
    # Staged direct-J/K materializes ERI tiles before contraction.  Fused kernels
    # avoid this term entirely.
    tile_roundtrip_bytes = 8 * int(eri_values) + int(tile_bytes_read)
    density_bytes = 8 * int(density_loads)
    output_payload_bytes = 8 * int(output_atomics)
    output_rmw_bytes_lower_bound = 16 * int(output_atomics)
    flops = int(fmuls) + int(reduction_adds) + int(output_atomics)

    payload_bytes = int(tile_bytes_read) + int(density_bytes) + int(output_payload_bytes)
    rmw_bytes = int(tile_bytes_read) + int(density_bytes) + int(output_rmw_bytes_lower_bound)
    ai_payload = float(flops) / float(payload_bytes) if payload_bytes > 0 else 0.0
    ai_rmw = float(flops) / float(rmw_bytes) if rmw_bytes > 0 else 0.0

    return DirectJKContractCost(
        contract_mode=str(mode),
        n_dm=int(n_dm_i),
        ntasks=int(ntasks),
        nA=int(nA_i),
        nB=int(nB_i),
        nC=int(nC_i),
        nD=int(nD_i),
        eri_values=int(eri_values),
        tile_loads=int(tile_loads),
        tile_bytes_read=int(tile_bytes_read),
        tile_roundtrip_bytes=int(tile_roundtrip_bytes),
        density_loads=int(density_loads),
        density_bytes=int(density_bytes),
        inner_fmuls=int(inner_fmuls),
        scale_fmuls=int(scale_fmuls),
        fmuls=int(fmuls),
        reduction_adds=int(reduction_adds),
        output_atomics=int(output_atomics),
        output_payload_bytes=int(output_payload_bytes),
        output_rmw_bytes_lower_bound=int(output_rmw_bytes_lower_bound),
        flops=int(flops),
        arithmetic_intensity_payload=float(ai_payload),
        arithmetic_intensity_rmw_lower_bound=float(ai_rmw),
    )




def estimate_direct_fock_contract_cost(
    *,
    histogram: np.ndarray,
    nA: int,
    nB: int,
    nC: int,
    nD: int,
    contract_mode: str,
    n_dm: int = 1,
) -> DirectJKContractCost:
    """Estimate cost for the direct Fock contraction kernels.

    The Fock kernels perform the same structural J/K work as the combined direct
    J/K kernels, but the K path carries an additional scale multiply by
    ``alpha = -0.5`` before the atomic updates.
    """

    base = estimate_direct_jk_contract_cost(
        histogram=histogram,
        nA=nA,
        nB=nB,
        nC=nC,
        nD=nD,
        contract_mode=contract_mode,
        want_J=True,
        want_K=True,
        n_dm=n_dm,
    )

    mode = str(contract_mode).strip().lower().replace("-", "_")
    hist = np.asarray(histogram, dtype=np.int64).reshape(-1)
    if hist.size != _HIST_SIZE:
        raise ValueError("histogram must have 8 bins")

    nA_i = int(nA)
    nB_i = int(nB)
    nC_i = int(nC)
    nD_i = int(nD)
    n_dm_i = int(n_dm)
    extra_k_scale = 0
    for idx, count_raw in enumerate(hist.tolist()):
        count = int(count_raw)
        if count <= 0:
            continue
        ab_neq = bool((idx >> 2) & 0x1)
        cd_neq = bool((idx >> 1) & 0x1)
        bra_ket_swap = bool(idx & 0x1)

        k_terms = 1 + int(cd_neq) + int(ab_neq) + int(ab_neq and cd_neq)
        if mode == "staged":
            m = int(count) * int(nA_i * nB_i) * int(nC_i * nD_i)
            extra_k_scale += int(n_dm_i) * int(m) * int(k_terms)
        else:
            k_outer = nA_i * nC_i
            if cd_neq:
                k_outer += nA_i * nD_i
            if ab_neq:
                k_outer += nB_i * nC_i
            if ab_neq and cd_neq:
                k_outer += nB_i * nD_i
            # Bra/ket swap reuses the scaled value for the mirrored atomic,
            # so the extra alpha multiply is per reduced outer result only.
            extra_k_scale += int(n_dm_i) * int(count) * int(k_outer)

    scale_fmuls = int(base.scale_fmuls) + int(extra_k_scale)
    fmuls = int(base.inner_fmuls) + int(scale_fmuls)
    flops = int(fmuls) + int(base.reduction_adds) + int(base.output_atomics)
    return replace(base, scale_fmuls=scale_fmuls, fmuls=fmuls, flops=flops)


def merge_contract_cost_row(row: dict, cost: DirectJKContractCost) -> None:
    """Accumulate a cost-model summary into a mutable stats row."""

    prefix = "contract_"
    prev_mode = str(row.get(f"{prefix}mode", "") or "")
    mode = str(cost.contract_mode)
    row[f"{prefix}mode"] = mode if (not prev_mode or prev_mode == mode) else "mixed"
    prev_n_dm = row.get(f"{prefix}n_dm")
    row[f"{prefix}n_dm"] = int(cost.n_dm) if prev_n_dm is None else int(prev_n_dm)
    for key, value in cost.as_dict().items():
        if key in {"contract_mode", "n_dm", "arithmetic_intensity_payload", "arithmetic_intensity_rmw_lower_bound"}:
            continue
        if isinstance(value, (int, np.integer)):
            row_key = f"{prefix}{key}"
            row[row_key] = int(row.get(row_key, 0)) + int(value)
    payload_num = float(row.get(f"{prefix}flops", 0))
    payload_den = float(
        row.get(f"{prefix}tile_bytes_read", 0)
        + row.get(f"{prefix}density_bytes", 0)
        + row.get(f"{prefix}output_payload_bytes", 0)
    )
    rmw_den = float(
        row.get(f"{prefix}tile_bytes_read", 0)
        + row.get(f"{prefix}density_bytes", 0)
        + row.get(f"{prefix}output_rmw_bytes_lower_bound", 0)
    )
    row[f"{prefix}arithmetic_intensity_payload"] = float(payload_num / payload_den) if payload_den > 0 else 0.0
    row[f"{prefix}arithmetic_intensity_rmw_lower_bound"] = float(payload_num / rmw_den) if rmw_den > 0 else 0.0


__all__ = [
    "DirectJKContractCost",
    "DirectJKSymmetryCase",
    "estimate_direct_jk_contract_cost",
    "estimate_direct_fock_contract_cost",
    "merge_contract_cost_row",
    "symmetry_histogram_from_cases",
    "symmetry_histogram_from_masks",
]
