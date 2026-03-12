from __future__ import annotations

"""Integral-direct J/K builder for SCF.

Builds Coulomb (J) and exchange (K) matrices by evaluating 4-center integrals
on-the-fly and contracting with the density matrix, without materializing the
full ERI tensor.  Memory is O(nao^2) GPU + O(ntasks) CPU for large systems.

For large systems (e.g. 100 H2O / 6-31g*) the number of screened shell-quartet
tasks can reach 1-10 billion.  This module avoids storing all tasks on GPU by
using *presorted slabs*: tasks are generated in Q-rank slabs on the GPU, sorted
by ERI class, and stored as either GPU-resident CuPy arrays (small systems) or
CPU numpy arrays (large systems).  Group offsets are computed on-GPU to avoid
expensive D→H of large class-ID arrays.

Uses specialized CUDA kernels (ssss, psss, pppp, dsds, ...) via the
``eri_dispatch`` infrastructure for optimal throughput, with automatic
bra/ket swap and fallback to generic Rys for unsupported angular momenta.

Usage::

    ctx = make_direct_jk_context(ao_basis, eps_schwarz=1e-12)
    J, K = direct_JK(ctx, D)
"""

import time
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from asuka.kernels import cueri as cueri_kernels


def _ppps_work_bucket_label(work: int) -> str:
    w = int(work)
    if w <= 8:
        return "1..8"
    if w <= 32:
        return "9..32"
    if w <= 128:
        return "33..128"
    return ">128"


_PPPS_PACKED_EXACT_SHAPES: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 3),
    (1, 6),
    (1, 9),
    (3, 1),
    (3, 3),
    (3, 6),
    (3, 9),
    (6, 1),
    (6, 3),
    (6, 6),
    (6, 9),
    (9, 1),
    (9, 3),
    (9, 6),
    (9, 9),
)


_PPPS_PACKED_SHAPE_TARGETS: tuple[str, ...] = (
    "1x1",
    "1x3",
    "1x6",
    "1x9",
    "3x1",
    "3x3",
    "3x6",
    "3x9",
    "6x1",
    "6x3",
    "6x6",
    "6x9",
    "9x1",
    "9x3",
    "9x6",
    "9x9",
)

_PPPS_PACKED_WORK_BUCKET_ORDER: tuple[str, ...] = ("1..8", "9..32", "33..128", ">128")
_PPPS_PREPACKED_CONTEXT_SHAPES: tuple[tuple[int, int], ...] = (
    (3, 3),
    (1, 3),
    (3, 1),
    (1, 1),
    (9, 3),
    (9, 1),
    (3, 9),
    (1, 9),
)


@dataclass(frozen=True)
class _PppsPackedGroup:
    shape_key: str
    work_bucket: str
    transpose: bool
    nAB: int
    nCD: int
    packed_start: int
    packed_stop: int
    original_task_idx: np.ndarray


@dataclass(frozen=True)
class _PppsPackedSlabPlan:
    packed_task_spAB: np.ndarray
    packed_task_spCD: np.ndarray
    packed_nAB: np.ndarray
    packed_nCD: np.ndarray
    packed_to_original: np.ndarray
    original_to_packed: np.ndarray
    groups: tuple[_PppsPackedGroup, ...]
    eligible_ntasks: int
    fallback_ntasks: int


@dataclass(frozen=True)
class _DirectJKPppsPackedGroup:
    task_idx: Any
    task_spAB: Any
    task_spCD: Any
    nAB: int
    nCD: int
    exact: bool
    ntasks: int
    packed: Any = None


@dataclass(frozen=True)
class _DirectJKPppsPackedExactData:
    Ax: Any
    Ay: Any
    Az: Any
    Bx: Any
    By: Any
    Bz: Any
    Cx: Any
    Cy: Any
    Cz: Any
    ab_eta: Any
    ab_Px: Any
    ab_Py: Any
    ab_Pz: Any
    ab_cK: Any
    cd_eta: Any
    cd_Qx: Any
    cd_Qy: Any
    cd_Qz: Any
    cd_cK: Any


def _shape_key_from_npair(nAB: int, nCD: int) -> str:
    return f"{int(nAB)}x{int(nCD)}"


def _build_ppps_packed_slab_plan(
    *,
    task_spAB: np.ndarray,
    task_spCD: np.ndarray,
    sp_npair: np.ndarray,
    transpose: bool,
    target_shape_keys: tuple[str, ...] = _PPPS_PACKED_SHAPE_TARGETS,
    max_tasks_per_group: int = 8192,
) -> _PppsPackedSlabPlan:
    task_spAB = np.asarray(task_spAB, dtype=np.int32).reshape(-1)
    task_spCD = np.asarray(task_spCD, dtype=np.int32).reshape(-1)
    if task_spAB.shape != task_spCD.shape:
        raise ValueError("task_spAB and task_spCD must have the same shape")
    if int(max_tasks_per_group) <= 0:
        raise ValueError("max_tasks_per_group must be > 0")

    ntasks = int(task_spAB.size)
    if ntasks == 0:
        empty_i32 = np.empty((0,), dtype=np.int32)
        return _PppsPackedSlabPlan(
            packed_task_spAB=empty_i32,
            packed_task_spCD=empty_i32,
            packed_nAB=empty_i32,
            packed_nCD=empty_i32,
            packed_to_original=empty_i32,
            original_to_packed=empty_i32,
            groups=tuple(),
            eligible_ntasks=0,
            fallback_ntasks=0,
        )

    sp_npair = np.asarray(sp_npair, dtype=np.int32).reshape(-1)
    nAB = np.asarray(sp_npair[task_spAB], dtype=np.int32)
    nCD = np.asarray(sp_npair[task_spCD], dtype=np.int32)
    shape_keys = [_shape_key_from_npair(int(abv), int(cdv)) for abv, cdv in zip(nAB.tolist(), nCD.tolist())]
    work_bins = [_ppps_work_bucket_label(int(abv) * int(cdv)) for abv, cdv in zip(nAB.tolist(), nCD.tolist())]

    target_rank = {str(key): i for i, key in enumerate(tuple(target_shape_keys))}
    work_rank = {str(key): i for i, key in enumerate(_PPPS_PACKED_WORK_BUCKET_ORDER)}

    eligible_idx: list[int] = []
    fallback_idx: list[int] = []
    eligible_keys: dict[tuple[str, str], list[int]] = {}
    fallback_keys: dict[tuple[str, str], list[int]] = {}
    fallback_seen_order: list[tuple[str, str]] = []

    for idx, (shape_key, work_bucket) in enumerate(zip(shape_keys, work_bins)):
        key = (str(shape_key), str(work_bucket))
        if shape_key in target_rank:
            eligible_idx.append(int(idx))
            eligible_keys.setdefault(key, []).append(int(idx))
        else:
            fallback_idx.append(int(idx))
            if key not in fallback_keys:
                fallback_keys[key] = []
                fallback_seen_order.append(key)
            fallback_keys[key].append(int(idx))

    ordered_keys = sorted(
        eligible_keys.keys(),
        key=lambda key: (target_rank.get(key[0], len(target_rank)), work_rank.get(key[1], len(work_rank)), key[0], key[1]),
    )
    ordered_keys.extend(fallback_seen_order)

    packed_perm_parts: list[np.ndarray] = []
    groups: list[_PppsPackedGroup] = []
    packed_start = 0
    for key in ordered_keys:
        if key in eligible_keys:
            idx_list = eligible_keys[key]
        else:
            idx_list = fallback_keys[key]
        idx_arr = np.asarray(idx_list, dtype=np.int32)
        if idx_arr.size == 0:
            continue
        shape_key, work_bucket = key
        first = int(idx_arr[0])
        group_nAB = int(nAB[first])
        group_nCD = int(nCD[first])
        for offset in range(0, int(idx_arr.size), int(max_tasks_per_group)):
            chunk = np.asarray(idx_arr[offset: offset + int(max_tasks_per_group)], dtype=np.int32)
            if chunk.size == 0:
                continue
            packed_perm_parts.append(chunk)
            packed_stop = packed_start + int(chunk.size)
            groups.append(
                _PppsPackedGroup(
                    shape_key=str(shape_key),
                    work_bucket=str(work_bucket),
                    transpose=bool(transpose),
                    nAB=int(group_nAB),
                    nCD=int(group_nCD),
                    packed_start=int(packed_start),
                    packed_stop=int(packed_stop),
                    original_task_idx=chunk,
                )
            )
            packed_start = int(packed_stop)

    packed_to_original = np.concatenate(packed_perm_parts).astype(np.int32, copy=False)
    original_to_packed = np.empty((ntasks,), dtype=np.int32)
    original_to_packed[packed_to_original] = np.arange(ntasks, dtype=np.int32)

    return _PppsPackedSlabPlan(
        packed_task_spAB=np.asarray(task_spAB[packed_to_original], dtype=np.int32),
        packed_task_spCD=np.asarray(task_spCD[packed_to_original], dtype=np.int32),
        packed_nAB=np.asarray(nAB[packed_to_original], dtype=np.int32),
        packed_nCD=np.asarray(nCD[packed_to_original], dtype=np.int32),
        packed_to_original=np.asarray(packed_to_original, dtype=np.int32),
        original_to_packed=original_to_packed,
        groups=tuple(groups),
        eligible_ntasks=int(len(eligible_idx)),
        fallback_ntasks=int(len(fallback_idx)),
    )


def _accumulate_ppps_packed_plan_stats(stats: dict[str, Any], plan: _PppsPackedSlabPlan) -> None:
    stats["family_ntasks"] = int(stats.get("family_ntasks", 0)) + int(plan.packed_to_original.size)
    stats["eligible_ntasks"] = int(stats.get("eligible_ntasks", 0)) + int(plan.eligible_ntasks)
    stats["fallback_ntasks"] = int(stats.get("fallback_ntasks", 0)) + int(plan.fallback_ntasks)
    stats["group_count"] = int(stats.get("group_count", 0)) + int(len(plan.groups))

    shape_counts = stats.setdefault("shape_counts", {})
    work_bin_counts = stats.setdefault("work_bin_counts", {})
    group_counts = stats.setdefault("group_counts", {})
    for group in plan.groups:
        nt = int(group.packed_stop) - int(group.packed_start)
        shape_counts[str(group.shape_key)] = int(shape_counts.get(str(group.shape_key), 0)) + int(nt)
        work_bin_counts[str(group.work_bucket)] = int(work_bin_counts.get(str(group.work_bucket), 0)) + int(nt)
        gkey = f"{group.shape_key}|{group.work_bucket}"
        group_counts[gkey] = int(group_counts.get(gkey, 0)) + int(nt)


def _build_direct_jk_ppps_packed_groups(
    *,
    task_spAB_dev,
    task_spCD_dev,
    task_spAB_cpu: np.ndarray,
    task_spCD_cpu: np.ndarray,
    sp_npair_cpu: np.ndarray,
    dsp=None,
    dbasis=None,
    pair_tables=None,
    exact_shapes: tuple[tuple[int, int], ...] = _PPPS_PREPACKED_CONTEXT_SHAPES,
):
    import cupy as cp  # noqa: PLC0415

    task_spAB_cpu = np.asarray(task_spAB_cpu, dtype=np.int32).reshape(-1)
    task_spCD_cpu = np.asarray(task_spCD_cpu, dtype=np.int32).reshape(-1)
    if task_spAB_cpu.shape != task_spCD_cpu.shape:
        raise ValueError("task_spAB_cpu and task_spCD_cpu must have the same shape")
    ntasks = int(task_spAB_cpu.size)
    if ntasks == 0:
        return tuple()

    sp_npair_cpu = np.asarray(sp_npair_cpu, dtype=np.int32).reshape(-1)
    nAB = np.asarray(sp_npair_cpu[task_spAB_cpu], dtype=np.int32)
    nCD = np.asarray(sp_npair_cpu[task_spCD_cpu], dtype=np.int32)
    exact_mask = np.zeros((ntasks,), dtype=bool)
    groups: list[_DirectJKPppsPackedGroup] = []

    for npair_ab, npair_cd in tuple(exact_shapes):
        idx = np.flatnonzero((nAB == int(npair_ab)) & (nCD == int(npair_cd))).astype(np.int32, copy=False)
        if int(idx.size) == 0:
            continue
        idx_dev = cp.asarray(idx, dtype=cp.int32)
        task_spAB_group = cp.ascontiguousarray(task_spAB_dev[idx_dev])
        task_spCD_group = cp.ascontiguousarray(task_spCD_dev[idx_dev])
        packed = None
        if dsp is not None and dbasis is not None and pair_tables is not None:
            packed = _pack_direct_jk_ppps_exact_group(
                task_spAB=task_spAB_group,
                task_spCD=task_spCD_group,
                dsp=dsp,
                dbasis=dbasis,
                pair_tables=pair_tables,
                nAB=int(npair_ab),
                nCD=int(npair_cd),
            )
        groups.append(
            _DirectJKPppsPackedGroup(
                task_idx=idx_dev,
                task_spAB=task_spAB_group,
                task_spCD=task_spCD_group,
                nAB=int(npair_ab),
                nCD=int(npair_cd),
                exact=True,
                ntasks=int(idx.size),
                packed=packed,
            )
        )
        exact_mask[idx] = True

    fallback_idx = np.flatnonzero(~exact_mask).astype(np.int32, copy=False)
    if int(fallback_idx.size) > 0:
        fallback_dev = cp.asarray(fallback_idx, dtype=cp.int32)
        groups.append(
            _DirectJKPppsPackedGroup(
                task_idx=fallback_dev,
                task_spAB=cp.ascontiguousarray(task_spAB_dev[fallback_dev]),
                task_spCD=cp.ascontiguousarray(task_spCD_dev[fallback_dev]),
                nAB=0,
                nCD=0,
                exact=False,
                ntasks=int(fallback_idx.size),
            )
        )

    return tuple(groups)


def _pack_direct_jk_ppps_exact_group(
    *,
    task_spAB,
    task_spCD,
    dsp,
    dbasis,
    pair_tables,
    nAB: int,
    nCD: int,
):
    import cupy as cp  # noqa: PLC0415

    A = dsp.sp_A[task_spAB]
    B = dsp.sp_B[task_spAB]
    C = dsp.sp_A[task_spCD]
    Ax = cp.ascontiguousarray(dbasis.shell_cx[A])
    Ay = cp.ascontiguousarray(dbasis.shell_cy[A])
    Az = cp.ascontiguousarray(dbasis.shell_cz[A])
    Bx = cp.ascontiguousarray(dbasis.shell_cx[B])
    By = cp.ascontiguousarray(dbasis.shell_cy[B])
    Bz = cp.ascontiguousarray(dbasis.shell_cz[B])
    Cx = cp.ascontiguousarray(dbasis.shell_cx[C])
    Cy = cp.ascontiguousarray(dbasis.shell_cy[C])
    Cz = cp.ascontiguousarray(dbasis.shell_cz[C])

    ab_base = dsp.sp_pair_start[task_spAB].astype(cp.int64, copy=False)
    cd_base = dsp.sp_pair_start[task_spCD].astype(cp.int64, copy=False)
    ab_offsets = cp.arange(int(nAB), dtype=cp.int64)[None, :]
    cd_offsets = cp.arange(int(nCD), dtype=cp.int64)[None, :]
    ab_idx = cp.ascontiguousarray((ab_base[:, None] + ab_offsets).reshape(-1))
    cd_idx = cp.ascontiguousarray((cd_base[:, None] + cd_offsets).reshape(-1))

    return _DirectJKPppsPackedExactData(
        Ax=Ax,
        Ay=Ay,
        Az=Az,
        Bx=Bx,
        By=By,
        Bz=Bz,
        Cx=Cx,
        Cy=Cy,
        Cz=Cz,
        ab_eta=cp.ascontiguousarray(pair_tables.pair_eta[ab_idx]),
        ab_Px=cp.ascontiguousarray(pair_tables.pair_Px[ab_idx]),
        ab_Py=cp.ascontiguousarray(pair_tables.pair_Py[ab_idx]),
        ab_Pz=cp.ascontiguousarray(pair_tables.pair_Pz[ab_idx]),
        ab_cK=cp.ascontiguousarray(pair_tables.pair_cK[ab_idx]),
        cd_eta=cp.ascontiguousarray(pair_tables.pair_eta[cd_idx]),
        cd_Qx=cp.ascontiguousarray(pair_tables.pair_Px[cd_idx]),
        cd_Qy=cp.ascontiguousarray(pair_tables.pair_Py[cd_idx]),
        cd_Qz=cp.ascontiguousarray(pair_tables.pair_Pz[cd_idx]),
        cd_cK=cp.ascontiguousarray(pair_tables.pair_cK[cd_idx]),
    )


def _slice_direct_jk_ppps_packed_exact_data(packed: _DirectJKPppsPackedExactData, idx, nAB: int, nCD: int):
    import cupy as cp  # noqa: PLC0415

    idx = cp.asarray(idx, dtype=cp.int32)
    def _sel1(x):
        return cp.ascontiguousarray(x[idx])
    def _sel2(x, width: int):
        return cp.ascontiguousarray(x.reshape((-1, int(width)))[idx].reshape(-1))
    return _DirectJKPppsPackedExactData(
        Ax=_sel1(packed.Ax),
        Ay=_sel1(packed.Ay),
        Az=_sel1(packed.Az),
        Bx=_sel1(packed.Bx),
        By=_sel1(packed.By),
        Bz=_sel1(packed.Bz),
        Cx=_sel1(packed.Cx),
        Cy=_sel1(packed.Cy),
        Cz=_sel1(packed.Cz),
        ab_eta=_sel2(packed.ab_eta, nAB),
        ab_Px=_sel2(packed.ab_Px, nAB),
        ab_Py=_sel2(packed.ab_Py, nAB),
        ab_Pz=_sel2(packed.ab_Pz, nAB),
        ab_cK=_sel2(packed.ab_cK, nAB),
        cd_eta=_sel2(packed.cd_eta, nCD),
        cd_Qx=_sel2(packed.cd_Qx, nCD),
        cd_Qy=_sel2(packed.cd_Qy, nCD),
        cd_Qz=_sel2(packed.cd_Qz, nCD),
        cd_cK=_sel2(packed.cd_cK, nCD),
    )


_PPPS_PREPACKED_RAW_KERNEL = None


def _get_ppps_prepacked_raw_kernel(cp):
    global _PPPS_PREPACKED_RAW_KERNEL
    if _PPPS_PREPACKED_RAW_KERNEL is not None:
        return _PPPS_PREPACKED_RAW_KERNEL
    code = r'''
extern "C" __device__ __forceinline__ void boys_f0_f1_f2_f3_f4(
    double T, double& F0, double& F1, double& F2, double& F3, double& F4) {
  if (T < 1e-8) {
    F0 = 1.0 - T / 3.0;
    F1 = 1.0 / 3.0 - T / 5.0;
    F2 = 1.0 / 5.0 - T / 7.0;
    F3 = 1.0 / 7.0 - T / 9.0;
    F4 = 1.0 / 9.0 - T / 11.0;
    return;
  }
  const double sqrtT = sqrt(T);
  F0 = 0.88622692545275801364908374167057 * erf(sqrtT) / sqrtT;
  const double e = exp(-T);
  F1 = (F0 - e) / (2.0 * T);
  F2 = (3.0 * F1 - e) / (2.0 * T);
  F3 = (5.0 * F2 - e) / (2.0 * T);
  F4 = (7.0 * F3 - e) / (2.0 * T);
}

extern "C" __device__ __forceinline__ double t3_component(
    int a, int b, int c, const double* dvec, double term_t3_f2, double term_t3_f3) {
  const double dab = (a == b) ? dvec[c] : 0.0;
  const double dac = (a == c) ? dvec[b] : 0.0;
  const double dbc = (b == c) ? dvec[a] : 0.0;
  return term_t3_f2 * (dab + dac + dbc) + term_t3_f3 * dvec[a] * dvec[b] * dvec[c];
}

extern "C" __global__ void ppps_prepacked_exact_scalar(
    const double* Ax, const double* Ay, const double* Az,
    const double* Bx, const double* By, const double* Bz,
    const double* Cx, const double* Cy, const double* Cz,
    const double* ab_eta, const double* ab_Px, const double* ab_Py, const double* ab_Pz, const double* ab_cK,
    const double* cd_eta, const double* cd_Qx, const double* cd_Qy, const double* cd_Qz, const double* cd_cK,
    int ntasks, int nAB, int nCD, double* eri_out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= ntasks) return;
  const double ax = Ax[t], ay = Ay[t], az = Az[t];
  const double bx = Bx[t], by = By[t], bz = Bz[t];
  const double cx = Cx[t], cy = Cy[t], cz = Cz[t];
  double acc[27];
#pragma unroll
  for (int i = 0; i < 27; ++i) acc[i] = 0.0;
  const int ab0 = t * nAB;
  const int cd0 = t * nCD;
  for (int ii = 0; ii < nAB; ++ii) {
    const int abi = ab0 + ii;
    const double p = ab_eta[abi];
    const double Px = ab_Px[abi];
    const double Py = ab_Py[abi];
    const double Pz = ab_Pz[abi];
    const double PA[3] = {Px - ax, Py - ay, Pz - az};
    const double PB[3] = {Px - bx, Py - by, Pz - bz};
    const double cKab = ab_cK[abi];
    for (int jj = 0; jj < nCD; ++jj) {
      const int cdj = cd0 + jj;
      const double q = cd_eta[cdj];
      const double Qx = cd_Qx[cdj];
      const double Qy = cd_Qy[cdj];
      const double Qz = cd_Qz[cdj];
      const double QC[3] = {Qx - cx, Qy - cy, Qz - cz};
      const double dvec[3] = {Px - Qx, Py - Qy, Pz - Qz};
      const double PQ2 = dvec[0]*dvec[0] + dvec[1]*dvec[1] + dvec[2]*dvec[2];
      const double denom = p + q;
      const double omega = p * q / denom;
      const double T = omega * PQ2;
      const double pref = 34.986836655249725 * rsqrt(denom) / (p * q);
      const double base = pref * cKab * cd_cK[cdj];
      double F0, F1, F2, F3, F4;
      boys_f0_f1_f2_f3_f4(T, F0, F1, F2, F3, F4);
      (void)F4;
      const double I = base * F0;
      const double omega_over_p = omega / p;
      const double omega_over_q = omega / q;
      const double Jp[3] = {
        -omega_over_p * base * F1 * dvec[0],
        -omega_over_p * base * F1 * dvec[1],
        -omega_over_p * base * F1 * dvec[2]};
      const double Jq[3] = {
         omega_over_q * base * F1 * dvec[0],
         omega_over_q * base * F1 * dvec[1],
         omega_over_q * base * F1 * dvec[2]};
      const double w2 = omega * omega;
      const double w3 = w2 * omega;
      const double inv4p2 = 1.0 / (4.0 * p * p);
      const double inv4pq = 1.0 / (4.0 * p * q);
      const double t4 = 4.0 * w2 * F2;
      const double t2 = 2.0 * omega * F1;
      double Kp[3][3];
      double L[3][3];
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          const double dij = (a == b) ? 1.0 : 0.0;
          const double H = base * (t4 * dvec[a] * dvec[b] - (a == b ? t2 : 0.0));
          Kp[a][b] = (H + 2.0 * p * I * dij) * inv4p2;
          L[a][b] = -H * inv4pq;
        }
      }
      const double term_t3_f2 = 4.0 * w2 * base * F2;
      const double term_t3_f3 = -8.0 * w3 * base * F3;
      for (int ia = 0; ia < 3; ++ia) {
        for (int ib = 0; ib < 3; ++ib) {
          const double a = PA[ia];
          const double b = PB[ib];
          const double dij = (ia == ib) ? 1.0 : 0.0;
          for (int ic = 0; ic < 3; ++ic) {
            const double cc = QC[ic];
            const double T3 = t3_component(ia, ib, ic, dvec, term_t3_f2, term_t3_f3);
            const double M_ijk = (-T3 + 4.0 * p * q * dij * Jq[ic]) / (8.0 * p * p * q);
            acc[ia * 9 + ib * 3 + ic] +=
              M_ijk + cc * Kp[ia][ib] + b * L[ia][ic] + b * cc * Jp[ia]
              + a * L[ib][ic] + a * cc * Jp[ib] + a * b * Jq[ic] + a * b * cc * I;
          }
        }
      }
    }
  }
  const int out0 = t * 27;
  for (int i = 0; i < 27; ++i) eri_out[out0 + i] = acc[i];
}
'''
    _PPPS_PREPACKED_RAW_KERNEL = cp.RawKernel(code, "ppps_prepacked_exact_scalar", options=("-std=c++17",))
    return _PPPS_PREPACKED_RAW_KERNEL


def _accumulate_ppps_shape_census(
    census: dict[str, Any],
    *,
    cls: str,
    transpose: bool,
    nAB: np.ndarray,
    nCD: np.ndarray,
) -> None:
    nAB = np.asarray(nAB, dtype=np.int64).reshape(-1)
    nCD = np.asarray(nCD, dtype=np.int64).reshape(-1)
    if nAB.shape != nCD.shape:
        raise ValueError("nAB and nCD must have the same shape")
    if nAB.size == 0:
        return

    census["family_ntasks"] = int(census.get("family_ntasks", 0)) + int(nAB.size)
    by_class = census.setdefault("by_class", {})
    by_class[str(cls)] = int(by_class.get(str(cls), 0)) + int(nAB.size)
    by_transpose = census.setdefault("by_transpose", {})
    tkey = "transpose" if bool(transpose) else "native"
    by_transpose[tkey] = int(by_transpose.get(tkey, 0)) + int(nAB.size)

    nAB_counts = census.setdefault("nAB_counts", {})
    uniq_ab, counts_ab = np.unique(nAB, return_counts=True)
    for val, cnt in zip(uniq_ab.tolist(), counts_ab.tolist()):
        key = str(int(val))
        nAB_counts[key] = int(nAB_counts.get(key, 0)) + int(cnt)

    nCD_counts = census.setdefault("nCD_counts", {})
    uniq_cd, counts_cd = np.unique(nCD, return_counts=True)
    for val, cnt in zip(uniq_cd.tolist(), counts_cd.tolist()):
        key = str(int(val))
        nCD_counts[key] = int(nCD_counts.get(key, 0)) + int(cnt)

    shape_counts = census.setdefault("shape_counts", {})
    uniq_shapes, counts_shape = np.unique(np.stack((nAB, nCD), axis=1), axis=0, return_counts=True)
    for (abv, cdv), cnt in zip(uniq_shapes.tolist(), counts_shape.tolist()):
        key = f"{int(abv)}x{int(cdv)}"
        shape_counts[key] = int(shape_counts.get(key, 0)) + int(cnt)

    work = nAB * nCD
    work_counts = census.setdefault("work_counts", {})
    uniq_work, counts_work = np.unique(work, return_counts=True)
    for val, cnt in zip(uniq_work.tolist(), counts_work.tolist()):
        key = str(int(val))
        work_counts[key] = int(work_counts.get(key, 0)) + int(cnt)

    work_bin_counts = census.setdefault("work_bin_counts", {})
    for val, cnt in zip(uniq_work.tolist(), counts_work.tolist()):
        key = _ppps_work_bucket_label(int(val))
        work_bin_counts[key] = int(work_bin_counts.get(key, 0)) + int(cnt)

    work_sum = int(work.sum(dtype=np.int64))
    census["total_work"] = int(census.get("total_work", 0)) + int(work_sum)
    work_min = int(work.min())
    work_max = int(work.max())
    if "work_min" not in census:
        census["work_min"] = int(work_min)
    else:
        census["work_min"] = min(int(census["work_min"]), int(work_min))
    if "work_max" not in census:
        census["work_max"] = int(work_max)
    else:
        census["work_max"] = max(int(census["work_max"]), int(work_max))


@dataclass(frozen=True)
class _SortedSlab:
    """Precomputed sorted task slab (shell-pair index pairs grouped by ERI class).

    If ``gpu_resident=True`` the arrays live on GPU (CuPy); otherwise they are
    CPU numpy arrays that are uploaded to GPU on each J/K call.
    """

    # Task arrays — either CuPy (gpu_resident) or numpy (cpu)
    ab_sorted: Any   # (ntasks_slab,) int32
    cd_sorted: Any   # (ntasks_slab,) int32
    ntasks: int
    # Class group metadata (always CPU numpy — tiny)
    class_ids: np.ndarray  # (nclass,) int32
    offsets: np.ndarray    # (nclass+1,) int32
    gpu_resident: bool


def _choose_n_bufs(nao: int, vram_budget_mb: int = 500) -> int:
    """Power-of-2 buffer count for multi-buffer Fock accumulation.

    Distributes atomicAdd contention across N independent Fock buffers.
    Bounded by VRAM budget so that ``N * nao^2 * 8`` bytes stays under *vram_budget_mb*.
    """
    buf_bytes = nao * nao * 8
    if buf_bytes <= 0:
        return 1
    max_bufs = max(1, int(vram_budget_mb * 1024**2 // buf_bytes))
    max_bufs = min(max_bufs, 128)  # cap at ~SM count
    n = 1
    while n * 2 <= max_bufs:
        n *= 2
    return n


@dataclass(frozen=True)
class _DirectJKGroupPlan:
    """Immutable per-class execution plan for a slab."""

    orig_cid: int
    kernel_cid: int
    class_label: str
    kernel_label: str
    transpose: bool
    hot_s1: bool
    j0: int
    j1: int
    nA: int
    nB: int
    nC: int
    nD: int
    ncomp: int
    chunk_ntasks: int
    use_warp_eri: bool
    use_warp_contract: bool


@dataclass(frozen=True)
class DirectJKContext:
    """Pre-computed, reusable context for integral-direct J/K builds."""

    ao_basis: Any
    nao: int
    max_l: int
    # GPU-resident basis data
    dbasis: Any
    dsp: Any
    pair_tables: Any
    sp_A_dev: Any           # cp.ndarray int32 (nsp,)
    sp_B_dev: Any           # cp.ndarray int32 (nsp,)
    shell_ao_start_dev: Any # cp.ndarray int32 (nshell,)
    sp_class_lo_dev: Any    # cp.ndarray int32 (nsp,)
    sp_class_lo_cpu: np.ndarray  # (nsp,) int32
    # Presorted task slabs
    slabs: tuple            # tuple[_SortedSlab, ...]
    plans: tuple            # tuple[tuple[_DirectJKGroupPlan, ...], ...] aligned with slabs
    ppps_packed_groups: tuple  # tuple[tuple[tuple[_DirectJKPppsPackedGroup, ...] | None, ...], ...]
    nsp: int
    ntasks: int
    eps_schwarz: float
    threads: int
    eval_threads: int
    contract_threads: int
    max_tile_bytes: int
    gpu_task_budget_bytes_used: int
    max_slab_tasks_used: int
    max_cpu_slab_ntasks: int
    # Schwarz bounds and shell geometry for density prescreening
    Q_sp: Any = None             # cp.ndarray float64 (nsp,) — Schwarz bounds per shell pair
    shell_l_dev: Any = None      # cp.ndarray int32 (nshell,) — angular momentum per shell


def _build_sorted_slab(
    perm32_dev,
    jmax_dev,
    sp_class_lo_dev,
    i0: int,
    i1: int,
    gpu_budget_bytes: int,
):
    """Generate, sort, and return a task slab for Q-ranks [i0, i1).

    Group offsets are computed on GPU (avoids D→H of large class-ID arrays).
    Task arrays are kept GPU-resident if they fit within ``gpu_budget_bytes``;
    otherwise they are transferred to CPU numpy.
    """
    import cupy as cp  # noqa: PLC0415

    n_slab = i1 - i0
    if n_slab <= 0:
        return None

    jmax_slab = jmax_dev[i0:i1]
    total = int(jmax_slab.sum())
    if total == 0:
        return None

    # --- Generate tasks on GPU ---
    group_offsets = cp.empty(n_slab + 1, dtype=cp.int64)
    group_offsets[0] = 0
    cp.cumsum(jmax_slab, out=group_offsets[1:])

    indices = cp.arange(total, dtype=cp.int64)
    k_arr = cp.searchsorted(group_offsets, indices, side="right") - 1
    j_arr = indices - group_offsets[k_arr]
    del group_offsets, jmax_slab

    ab = perm32_dev[k_arr + cp.int64(i0)]
    cd = perm32_dev[j_arr]
    del k_arr, j_arr, indices

    swap = cd > ab
    ab_out = cp.where(swap, cd, ab)
    cd_out = cp.where(swap, ab, cd)
    del ab, cd, swap

    class_id_dev = sp_class_lo_dev[ab_out] | (sp_class_lo_dev[cd_out] << cp.int32(16))
    perm_dev = cp.argsort(class_id_dev, kind="stable")
    ab_sorted_dev = cp.ascontiguousarray(ab_out[perm_dev])
    cd_sorted_dev = cp.ascontiguousarray(cd_out[perm_dev])
    class_id_sorted_dev = class_id_dev[perm_dev]
    del ab_out, cd_out, class_id_dev, perm_dev

    # --- Compute group offsets on GPU (avoids D→H of full class_id array) ---
    slab_nt = total
    if slab_nt > 1:
        diff = class_id_sorted_dev[1:] != class_id_sorted_dev[:-1]
        change_positions = cp.nonzero(diff)[0] + 1  # positions where class changes
        del diff
        # D→H only the small arrays: change_positions (nclass-1,) and first/last class IDs
        change_cpu = cp.asnumpy(change_positions).astype(np.int32)
        del change_positions
    else:
        change_cpu = np.empty((0,), dtype=np.int32)

    # class IDs at group starts: D→H only nclass values (tiny)
    if change_cpu.size == 0:
        group_start_positions = cp.zeros(1, dtype=cp.int64)
    else:
        group_start_positions = cp.concatenate([
            cp.zeros(1, dtype=cp.int64),
            cp.asarray(change_cpu.astype(np.int64)),
        ])
    class_ids_dev_small = class_id_sorted_dev[cp.asarray(group_start_positions, dtype=cp.int64)]
    class_ids_cpu = cp.asnumpy(class_ids_dev_small).astype(np.int32)
    del class_id_sorted_dev, class_ids_dev_small, group_start_positions

    offsets_cpu = np.concatenate(([0], change_cpu, [slab_nt])).astype(np.int32)
    del change_cpu

    # --- Decide storage: GPU or CPU ---
    task_bytes = total * 2 * 4  # ab + cd as int32
    if task_bytes <= gpu_budget_bytes:
        # GPU-resident: keep CuPy arrays (zero per-iteration H→D)
        return _SortedSlab(
            ab_sorted=ab_sorted_dev,
            cd_sorted=cd_sorted_dev,
            ntasks=int(total),
            class_ids=class_ids_cpu,
            offsets=offsets_cpu,
            gpu_resident=True,
        )
    else:
        # CPU-resident: D→H task arrays (paid once per context build, not per iteration).
        # Use pinned (page-locked) host memory for faster H→D transfers.
        ab_cpu = cp.asnumpy(ab_sorted_dev).astype(np.int32)
        cd_cpu = cp.asnumpy(cd_sorted_dev).astype(np.int32)
        del ab_sorted_dev, cd_sorted_dev
        try:
            ab_pinned = cp.cuda.alloc_pinned_memory(ab_cpu.nbytes)
            cd_pinned = cp.cuda.alloc_pinned_memory(cd_cpu.nbytes)
            ab_host = np.frombuffer(ab_pinned, dtype=np.int32, count=ab_cpu.size)
            cd_host = np.frombuffer(cd_pinned, dtype=np.int32, count=cd_cpu.size)
            ab_host[:] = ab_cpu
            cd_host[:] = cd_cpu
            ab_cpu = ab_host
            cd_cpu = cd_host
        except Exception:
            pass  # fall back to regular (pageable) numpy arrays
        return _SortedSlab(
            ab_sorted=ab_cpu,
            cd_sorted=cd_cpu,
            ntasks=int(total),
            class_ids=class_ids_cpu,
            offsets=offsets_cpu,
            gpu_resident=False,
        )


@dataclass
class DirectJKWorkspace:
    """Per-device persistent workspace for direct J/K builds."""

    copy_stream: Any
    upload_ab: list[Any] | None
    upload_cd: list[Any] | None
    upload_done: list[Any]
    compute_done: list[Any]
    max_ntasks: int
    D_flat: Any | None
    J_flat: Any | None
    K_flat: Any | None
    J_bufs: Any | None
    K_bufs: Any | None
    max_nao2: int
    max_n_bufs: int


_DIRECT_JK_WS_BY_DEVICE: dict[int, DirectJKWorkspace | None] = {}


_DIRECT_JK_POLICY_ENGINES = frozenset(
    {
        "default",
        "staged_block",
        "staged_warp",
        "staged_warp_eri",
        "staged_warp_contract",
        "fused_jk",
    }
)


def _normalize_direct_jk_engine(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower().replace("-", "_")
    if not s:
        return None
    aliases = {
        "block": "staged_block",
        "staged": "staged_block",
        "warp": "staged_warp",
        "warp_eri": "staged_warp_eri",
        "eri_warp": "staged_warp_eri",
        "warp_contract": "staged_warp_contract",
        "contract_warp": "staged_warp_contract",
        "fused": "fused_jk",
    }
    s = aliases.get(s, s)
    if s not in _DIRECT_JK_POLICY_ENGINES:
        return None
    return str(s)


def _default_direct_jk_class_policy(*, max_l: int, large_workload: bool) -> dict[str, str]:
    del max_l, large_workload
    return {"ssss": "fused_jk"}


def _parse_direct_jk_class_policy_env(raw: str | None) -> dict[str, str]:
    text = str(raw or "").strip()
    if not text:
        return {}
    out: dict[str, str] = {}
    for item in text.replace(";", ",").split(","):
        token = str(item).strip()
        if not token:
            continue
        if "=" not in token:
            continue
        cls_raw, eng_raw = token.split("=", 1)
        cls = str(cls_raw).strip().lower()
        eng = _normalize_direct_jk_engine(eng_raw)
        if cls and eng is not None:
            out[cls] = eng
    return out


def release_direct_jk_workspace_cache() -> None:
    """Release cached direct-JK workspaces."""

    try:
        _DIRECT_JK_WS_BY_DEVICE.clear()
    except Exception:
        pass


def _get_direct_jk_workspace(cp, ctx: DirectJKContext) -> DirectJKWorkspace | None:
    need = int(getattr(ctx, "max_cpu_slab_ntasks", 0) or 0)
    dev = int(cp.cuda.runtime.getDevice())
    ws = _DIRECT_JK_WS_BY_DEVICE.get(dev)
    if ws is not None and int(ws.max_ntasks) >= need:
        return ws

    # (Re)allocate upload buffers sized to the largest CPU slab in this context.
    copy_stream = cp.cuda.Stream(non_blocking=True)
    if need > 0:
        upload_ab = [cp.empty((need,), dtype=cp.int32), cp.empty((need,), dtype=cp.int32)]
        upload_cd = [cp.empty((need,), dtype=cp.int32), cp.empty((need,), dtype=cp.int32)]
    else:
        upload_ab = None
        upload_cd = None
    upload_done = [cp.cuda.Event(), cp.cuda.Event()]
    compute_done = [cp.cuda.Event(), cp.cuda.Event()]
    # Mark buffers as initially reusable.
    cur = cp.cuda.get_current_stream()
    compute_done[0].record(cur)
    compute_done[1].record(cur)

    ws = DirectJKWorkspace(
        copy_stream=copy_stream,
        upload_ab=upload_ab,
        upload_cd=upload_cd,
        upload_done=upload_done,
        compute_done=compute_done,
        max_ntasks=int(need),
        D_flat=None,
        J_flat=None,
        K_flat=None,
        J_bufs=None,
        K_bufs=None,
        max_nao2=0,
        max_n_bufs=0,
    )
    _DIRECT_JK_WS_BY_DEVICE[dev] = ws
    return ws


def _ensure_direct_jk_matrix_buffers(cp, ws: DirectJKWorkspace | None, *, nao: int, n_bufs: int, want_J: bool, want_K: bool):
    if ws is None:
        return None, None, None, None, None

    nao2 = int(nao) * int(nao)
    if int(ws.max_nao2) < nao2 or ws.D_flat is None or ws.J_flat is None or ws.K_flat is None:
        ws.D_flat = cp.empty((nao2,), dtype=cp.float64)
        ws.J_flat = cp.empty((nao2,), dtype=cp.float64)
        ws.K_flat = cp.empty((nao2,), dtype=cp.float64)
        ws.max_nao2 = int(nao2)

    D_flat = ws.D_flat[:nao2]
    J_flat = ws.J_flat[:nao2] if want_J else None
    K_flat = ws.K_flat[:nao2] if want_K else None
    return D_flat, J_flat, K_flat, None, None


def _auto_direct_jk_budgets(cp, *, max_slab_tasks: int | None, gpu_task_budget_bytes: int | None) -> tuple[int, int]:
    """Return (max_slab_tasks, gpu_task_budget_bytes), filling autos for None."""

    # Historical constants remain the defaults; passing None opts into auto.
    max_slab_tasks_v = int(200_000_000) if max_slab_tasks is None else max(1, int(max_slab_tasks))
    gpu_task_budget_bytes_v = int(2 << 30) if gpu_task_budget_bytes is None else max(0, int(gpu_task_budget_bytes))

    if max_slab_tasks is not None and gpu_task_budget_bytes is not None:
        return max_slab_tasks_v, gpu_task_budget_bytes_v

    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        free_bytes = int(free_bytes)
        total_bytes = int(total_bytes)
    except Exception:
        return max_slab_tasks_v, gpu_task_budget_bytes_v

    # Headroom for matrices, pair tables, CuPy pool, and other phases.
    headroom = max(2 << 30, int(0.10 * float(total_bytes)))
    usable = max(0, free_bytes - int(headroom))

    if gpu_task_budget_bytes is None:
        gpu_task_budget_bytes_v = max(0, min(int(usable), int(0.45 * float(usable))))

    if max_slab_tasks is None:
        # Slab build is argsort-heavy and allocates several int64 temporaries.
        # Keep a conservative scratch budget per slab.
        scratch_budget = min(int(0.25 * float(usable)), 6 << 30)
        upload_budget = min(int(0.10 * float(usable)), 3 << 30)
        # Conservative per-task scratch model (~64 B/task) + double-buffer upload (16 B/task).
        from_scratch = int(max(1, scratch_budget // 64))
        from_upload = int(max(1, upload_budget // 16))
        max_slab_tasks_v = int(min(200_000_000, max(5_000_000, min(from_scratch, from_upload))))

    return int(max_slab_tasks_v), int(gpu_task_budget_bytes_v)


def make_direct_jk_context(
    ao_basis,
    *,
    eps_schwarz: float = 1e-12,
    threads: int = 256,
    max_tile_bytes: int = 256 << 20,
    max_slab_tasks: int | None = 200_000_000,
    gpu_task_budget_bytes: int | None = 2 << 30,
) -> DirectJKContext:
    """One-time setup: build shell pairs, Schwarz bounds, and presorted task slabs.

    Parameters
    ----------
    ao_basis
        Packed AO basis object (Cartesian).
    eps_schwarz
        Schwarz screening threshold.
    threads
        CUDA threads per block.
    max_tile_bytes
        Maximum bytes for the integral tile buffer per chunk.
    max_slab_tasks
        Maximum tasks per slab (bounds peak GPU memory during context build).
        Pass None for an auto-selected value based on available VRAM.
    gpu_task_budget_bytes
        If a slab's task arrays fit within this budget (default 2 GB), they
        are kept GPU-resident for zero per-iteration H→D overhead.
        Pass None for an auto-selected value based on available VRAM.
    """

    import cupy as cp  # noqa: PLC0415

    from asuka.cueri.gpu import (  # noqa: PLC0415
        CUDA_MAX_L,
        CUDA_MAX_NROOTS,
        build_pair_tables_ss_device,
        has_cuda_ext,
        to_device_basis_ss,
        to_device_shell_pairs,
    )
    from asuka.cueri.shell_pairs import build_shell_pairs_l_order  # noqa: PLC0415
    from asuka.integrals.int1e_cart import nao_cart_from_basis  # noqa: PLC0415

    if not has_cuda_ext():
        raise RuntimeError("cuERI CUDA extension not available")

    nao = int(nao_cart_from_basis(ao_basis))
    if nao <= 0:
        raise ValueError("AO basis has zero AO functions")

    shell_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    if shell_l.size == 0:
        raise ValueError("AO basis has zero shells")
    max_l = int(shell_l.max())
    if max_l > CUDA_MAX_L:
        raise NotImplementedError(
            f"CUDA direct J/K supports only l<={CUDA_MAX_L} (nroots<={CUDA_MAX_NROOTS})"
        )

    max_slab_tasks_i, gpu_task_budget_bytes_i = _auto_direct_jk_budgets(
        cp,
        max_slab_tasks=max_slab_tasks,
        gpu_task_budget_bytes=gpu_task_budget_bytes,
    )

    sp = build_shell_pairs_l_order(ao_basis)
    nsp = int(sp.sp_A.shape[0])
    if nsp == 0:
        raise ValueError("No shell pairs generated")

    eps_f = float(eps_schwarz)
    if eps_f > 0.0:
        from asuka.cueri.screening import schwarz_shellpairs_device  # noqa: PLC0415

        Q_dev = schwarz_shellpairs_device(
            ao_basis, sp,
            threads=int(threads),
            max_tiles_bytes=int(max_tile_bytes),
        )
        Q_np = cp.asnumpy(Q_dev)
        Q_sp_dev = Q_dev  # keep for density prescreening
    else:
        Q_np = np.ones((nsp,), dtype=np.float64)
        Q_sp_dev = None  # populated below when building context

    perm = np.argsort(-Q_np, kind="stable")
    Q_sorted = Q_np[perm]
    n_valid = int(np.searchsorted(-Q_sorted, 0.0, side="left"))
    if n_valid == 0:
        raise ValueError("All shell pairs have zero Schwarz bound")

    Q_sorted = Q_sorted[:n_valid]
    neg_Q_sorted = -Q_sorted
    perm32_cpu = perm[:n_valid].astype(np.int32)

    if eps_f > 0.0:
        thrs = eps_f / np.maximum(Q_sorted, 1e-300)
        jmax_uncapped = np.searchsorted(neg_Q_sorted, -thrs, side="right").astype(np.int64)
        jmax_cpu = np.minimum(jmax_uncapped, np.arange(n_valid, dtype=np.int64) + 1)
    else:
        jmax_cpu = np.arange(1, n_valid + 1, dtype=np.int64)

    ntasks = int(jmax_cpu.sum())
    cumtasks_cpu = np.concatenate(([np.int64(0)], np.cumsum(jmax_cpu)))

    sp_A_np = np.asarray(sp.sp_A, dtype=np.int32)
    sp_B_np = np.asarray(sp.sp_B, dtype=np.int32)
    la_sp = shell_l[sp_A_np]
    lb_sp = shell_l[sp_B_np]
    sp_class_lo_cpu = ((la_sp & 0xFF) | ((lb_sp & 0xFF) << 8)).astype(np.int32)

    sp_class_lo_dev = cp.ascontiguousarray(cp.asarray(sp_class_lo_cpu, dtype=cp.int32))
    perm32_dev = cp.ascontiguousarray(cp.asarray(perm32_cpu, dtype=cp.int32))
    jmax_dev = cp.ascontiguousarray(cp.asarray(jmax_cpu, dtype=cp.int64))

    # Build presorted slabs with cumulative GPU budget tracking
    slabs = []
    gpu_bytes_used = 0
    max_cpu_slab_ntasks = 0
    slab_i0 = 0
    while slab_i0 < n_valid:
        target = cumtasks_cpu[slab_i0] + int(max_slab_tasks_i)
        slab_i1 = int(np.searchsorted(cumtasks_cpu, target, side="right")) - 1
        slab_i1 = max(slab_i0 + 1, slab_i1)
        slab_i1 = min(slab_i1, n_valid)

        remaining_gpu = max(0, int(gpu_task_budget_bytes_i) - int(gpu_bytes_used))
        slab = _build_sorted_slab(
            perm32_dev, jmax_dev, sp_class_lo_dev,
            slab_i0, slab_i1,
            gpu_budget_bytes=remaining_gpu,
        )
        if slab is not None:
            if slab.gpu_resident:
                gpu_bytes_used += slab.ab_sorted.nbytes + slab.cd_sorted.nbytes
            else:
                max_cpu_slab_ntasks = max(int(max_cpu_slab_ntasks), int(slab.ntasks))
            slabs.append(slab)

        slab_i0 = slab_i1

    del perm32_dev, jmax_dev
    # Release slab-build temporaries from pool to avoid fragmentation during J/K
    # (only non-resident blocks; GPU-resident slab arrays are kept)
    cp.get_default_memory_pool().free_all_blocks()

    dbasis = to_device_basis_ss(ao_basis)
    dsp = to_device_shell_pairs(sp)
    pair_tables = build_pair_tables_ss_device(dbasis, dsp, stream=None, threads=int(threads))

    sp_A_dev = cp.ascontiguousarray(cp.asarray(sp_A_np, dtype=cp.int32))
    sp_B_dev = cp.ascontiguousarray(cp.asarray(sp_B_np, dtype=cp.int32))
    shell_ao_start_np = np.asarray(ao_basis.shell_ao_start, dtype=np.int32).ravel()
    shell_ao_start_dev = cp.ascontiguousarray(cp.asarray(shell_ao_start_np, dtype=cp.int32))

    # Precompute a per-slab execution plan so the hot loop is a simple executor.
    from asuka.cueri.cart import ncart  # noqa: PLC0415
    from asuka.cueri.eri_dispatch import resolve_kernel_class_id  # noqa: PLC0415
    from asuka.cueri.tasks import decode_eri_class_id  # noqa: PLC0415

    def _env_int_or_none(name: str) -> int | None:
        v = os.environ.get(str(name), None)
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        try:
            return int(s)
        except Exception:
            return None

    def _env_flag(name: str, default: bool = False) -> bool:
        v = os.environ.get(str(name), None)
        if v is None:
            return bool(default)
        s = str(v).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        return bool(default)

    def _sanitize_threads(v: int, fallback: int) -> int:
        try:
            x = int(v)
        except Exception:
            x = int(fallback)
        x = max(32, min(1024, int(x)))
        x = int(x // 32) * 32
        if x <= 0:
            x = int(fallback)
        return max(32, min(1024, int(x)))

    large_workload = bool(int(ntasks) >= 1_000_000)

    # Kernel launch geometry knobs:
    # - eval threads for ERI class kernels
    # - contract threads for dense scatter/contract kernels
    # For large screened workloads, 128-thread eval/contract launch geometry is
    # typically better than inheriting a larger generic block size.
    # Keep an env override for reproducibility and local tuning.
    auto_threads = _env_flag("ASUKA_DIRECT_JK_THREADS_AUTO", default=True)
    eval_threads_env = _env_int_or_none("ASUKA_DIRECT_JK_EVAL_THREADS")
    contract_threads_env = _env_int_or_none("ASUKA_DIRECT_JK_CONTRACT_THREADS")
    if eval_threads_env is None:
        eval_threads = _sanitize_threads(
            min(int(threads), 128) if (auto_threads and large_workload) else int(threads),
            int(threads),
        )
    else:
        eval_threads = _sanitize_threads(int(eval_threads_env), int(threads))
    if contract_threads_env is None:
        contract_threads = _sanitize_threads(
            min(int(threads), 128) if (auto_threads and large_workload) else int(threads),
            int(threads),
        )
    else:
        contract_threads = _sanitize_threads(int(contract_threads_env), int(threads))

    # Warp kernels are beneficial for some tiny cases but can be slower on
    # large resident task sets due to reduced per-block work / launch shape.
    # Keep behavior tunable and use a workload-aware default.
    warp_eri_ncomp_max_env = _env_int_or_none("ASUKA_DIRECT_JK_WARP_ERI_NCOMP_MAX")
    warp_contract_ncomp_max_env = _env_int_or_none("ASUKA_DIRECT_JK_WARP_CONTRACT_NCOMP_MAX")
    if warp_eri_ncomp_max_env is None:
        if large_workload and int(max_l) <= 2:
            warp_eri_ncomp_max = 108
        elif large_workload:
            warp_eri_ncomp_max = 128
        else:
            warp_eri_ncomp_max = 128
    else:
        warp_eri_ncomp_max = max(0, int(warp_eri_ncomp_max_env))
    if warp_contract_ncomp_max_env is None:
        warp_contract_ncomp_max = 0 if large_workload else 128
    else:
        warp_contract_ncomp_max = max(0, int(warp_contract_ncomp_max_env))

    def _am_label(l: int) -> str:
        am = "spdfghijklm"
        li = int(l)
        if 0 <= li < len(am):
            return am[li]
        return f"l{li}"

    def _class_label_from_cid(cid: int) -> str:
        x = int(cid) & 0xFFFFFFFF
        la = x & 0xFF
        lb = (x >> 8) & 0xFF
        lc = (x >> 16) & 0xFF
        ld = (x >> 24) & 0xFF
        return f"{_am_label(la)}{_am_label(lb)}{_am_label(lc)}{_am_label(ld)}"

    plans: list[tuple[_DirectJKGroupPlan, ...]] = []
    for slab in slabs:
        slab_plans: list[_DirectJKGroupPlan] = []
        for g in range(int(slab.class_ids.shape[0])):
            orig_cid = int(slab.class_ids[g])
            j0 = int(slab.offsets[g])
            j1 = int(slab.offsets[g + 1])
            la, lb, lc, ld = decode_eri_class_id(orig_cid)
            kernel_cid, transpose = resolve_kernel_class_id(orig_cid)
            nA = int(ncart(int(la)))
            nB = int(ncart(int(lb)))
            nC = int(ncart(int(lc)))
            nD = int(ncart(int(ld)))
            ncomp = int(nA * nB * nC * nD)
            chunk_ntasks = int(max(1, int(max_tile_bytes) // max(int(ncomp) * 8, 1)))
            use_warp_eri = bool(ncomp <= int(warp_eri_ncomp_max))
            use_warp_contract = bool(ncomp <= int(warp_contract_ncomp_max))
            orig_label = _class_label_from_cid(int(orig_cid))
            kernel_label = _class_label_from_cid(int(kernel_cid))
            hot_s1 = bool(
                (int(nB) == 1 and int(nD) == 1)
                and (int(nA) in (1, 3, 6))
                and (int(nC) in (1, 3, 6))
            )
            slab_plans.append(
                _DirectJKGroupPlan(
                    orig_cid=int(orig_cid),
                    kernel_cid=int(kernel_cid),
                    class_label=str(orig_label),
                    kernel_label=str(kernel_label),
                    transpose=bool(transpose),
                    hot_s1=bool(hot_s1),
                    j0=int(j0),
                    j1=int(j1),
                    nA=int(nA),
                    nB=int(nB),
                    nC=int(nC),
                    nD=int(nD),
                    ncomp=int(ncomp),
                    chunk_ntasks=int(chunk_ntasks),
                    use_warp_eri=bool(use_warp_eri),
                    use_warp_contract=bool(use_warp_contract),
                )
            )
        plans.append(tuple(slab_plans))

    return DirectJKContext(
        ao_basis=ao_basis,
        nao=int(nao),
        max_l=int(max_l),
        dbasis=dbasis,
        dsp=dsp,
        pair_tables=pair_tables,
        sp_A_dev=sp_A_dev,
        sp_B_dev=sp_B_dev,
        shell_ao_start_dev=shell_ao_start_dev,
        sp_class_lo_dev=sp_class_lo_dev,
        sp_class_lo_cpu=sp_class_lo_cpu,
        slabs=tuple(slabs),
        plans=tuple(plans),
        ppps_packed_groups=tuple(),
        nsp=int(nsp),
        ntasks=int(ntasks),
        eps_schwarz=float(eps_f),
        threads=int(threads),
        eval_threads=int(eval_threads),
        contract_threads=int(contract_threads),
        max_tile_bytes=int(max_tile_bytes),
        gpu_task_budget_bytes_used=int(gpu_task_budget_bytes_i),
        max_slab_tasks_used=int(max_slab_tasks_i),
        max_cpu_slab_ntasks=int(max_cpu_slab_ntasks),
        Q_sp=Q_sp_dev if eps_f > 0.0 else cp.ones((nsp,), dtype=cp.float64),
        shell_l_dev=cp.asarray(shell_l, dtype=cp.int32),
    )


def _compute_D_sp_max(D_gpu, sp_A_dev, sp_B_dev, shell_ao_start_dev, shell_l_dev, nsp, cp):
    """Compute max|D[μ,ν]| per shell pair on GPU.

    For each shell pair (A, B), compute the maximum absolute density matrix
    element over all AO pairs (μ ∈ A, ν ∈ B).  Returns a (nsp,) float64 array.
    """
    nao = D_gpu.shape[0]
    # Build shell-pair → max|D| mapping via CuPy RawKernel
    D_sp = cp.empty((nsp,), dtype=cp.float64)
    if nsp == 0:
        return D_sp
    _kernel_code = r"""
extern "C" __global__ void compute_D_sp_max(
    const double* __restrict__ D,
    const int* __restrict__ sp_A,
    const int* __restrict__ sp_B,
    const int* __restrict__ shell_ao_start,
    const int* __restrict__ shell_l,
    int nsp, int nao,
    double* __restrict__ D_sp) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nsp) return;
  int sA = sp_A[idx];
  int sB = sp_B[idx];
  int a0 = shell_ao_start[sA];
  int b0 = shell_ao_start[sB];
  int lA = shell_l[sA];
  int lB = shell_l[sB];
  int nA = (lA + 1) * (lA + 2) / 2;
  int nB = (lB + 1) * (lB + 2) / 2;
  double mx = 0.0;
  for (int ia = 0; ia < nA; ia++) {
    for (int ib = 0; ib < nB; ib++) {
      double val = D[(a0 + ia) * nao + (b0 + ib)];
      double abs_val = val < 0.0 ? -val : val;
      if (abs_val > mx) mx = abs_val;
      // Also check symmetric element for upper-triangle shell pairs
      if (sA != sB) {
        double val2 = D[(b0 + ib) * nao + (a0 + ia)];
        double abs_val2 = val2 < 0.0 ? -val2 : val2;
        if (abs_val2 > mx) mx = abs_val2;
      }
    }
  }
  D_sp[idx] = mx;
}
"""
    _kern = cp.RawKernel(_kernel_code, "compute_D_sp_max")
    threads = 256
    blocks = (nsp + threads - 1) // threads
    _kern((blocks,), (threads,), (
        D_gpu, sp_A_dev, sp_B_dev, shell_ao_start_dev, shell_l_dev,
        np.int32(nsp), np.int32(nao), D_sp,
    ))
    return D_sp


def direct_JK(
    ctx: DirectJKContext,
    D,
    *,
    want_J: bool = True,
    want_K: bool = True,
    profile: dict | None = None,
    stats: dict | None = None,
):
    """Build J and K via streaming integral-direct 4-center evaluation.

    GPU-resident slabs (small systems) incur zero H→D overhead.
    CPU-resident slabs (large systems) are streamed to GPU one at a time.

    Parameters
    ----------
    ctx
        Pre-built context from :func:`make_direct_jk_context`.
    D
        AO density matrix, shape ``(nao, nao)``, CuPy or NumPy array.
    want_J, want_K
        Which matrices to compute.
    profile
        Optional dict for timing statistics.  **Note**: profile mode inserts
        ``event.synchronize()`` after each contraction kernel call, serializing
        execution.  Reported timings are not representative of production
        throughput.

    Returns
    -------
    (J, K) : tuple
        Coulomb and exchange matrices, each ``(nao, nao)`` CuPy array, or None.
    """

    if not want_J and not want_K:
        return None, None

    import cupy as cp  # noqa: PLC0415

    _ext = cueri_kernels.require_ext()
    from asuka.cueri.eri_dispatch import KernelBatch, run_kernel_batch_spd  # noqa: PLC0415
    from asuka.cueri.tasks import TaskList  # noqa: PLC0415

    nao = ctx.nao
    base_threads = int(ctx.threads)
    eval_threads = int(getattr(ctx, "eval_threads", base_threads))
    contract_threads = int(getattr(ctx, "contract_threads", base_threads))

    D_gpu = cp.asarray(D, dtype=cp.float64)
    if D_gpu.ndim != 2 or D_gpu.shape != (nao, nao):
        raise ValueError(f"D must be ({nao}, {nao}), got {tuple(D_gpu.shape)}")

    # Density prescreening: compute max|D| per shell pair for CSAM screening
    eps_density_env = os.environ.get("ASUKA_DIRECT_JK_EPS_DENSITY", "")
    eps_density = float(eps_density_env) if eps_density_env.strip() else max(ctx.eps_schwarz, 1e-10)
    Q_sp = getattr(ctx, "Q_sp", None)
    shell_l_dev = getattr(ctx, "shell_l_dev", None)
    D_sp_dev = None
    if Q_sp is not None and shell_l_dev is not None and eps_density > 0:
        D_sp_dev = _compute_D_sp_max(
            D_gpu, ctx.sp_A_dev, ctx.sp_B_dev, ctx.shell_ao_start_dev,
            shell_l_dev, ctx.nsp, cp,
        )

    compute_stream = cp.cuda.get_current_stream()
    stream_ptr = int(compute_stream.ptr)
    ws = _get_direct_jk_workspace(cp, ctx)
    n_kernel_calls = 0
    t0 = time.perf_counter()
    hot_s1_env = str(os.environ.get("ASUKA_DENSE_CONTRACT_HOT_S1", "") or "").strip().lower()
    hot_s1_contract_enabled = hot_s1_env not in {"0", "false", "off", "no"}
    eri_block_pref_env = str(os.environ.get("ASUKA_DIRECT_JK_ERI_BLOCK_MODE", "") or "").strip().lower()
    ppps_census_env = str(os.environ.get("ASUKA_DIRECT_JK_PPPS_CENSUS", "") or "").strip().lower()
    ppps_census_enabled = ppps_census_env in {"1", "true", "on", "yes"}
    ppps_packed_plan_env = str(os.environ.get("ASUKA_DIRECT_JK_PPPS_PACKED_PLAN", "") or "").strip().lower()
    ppps_packed_plan_enabled = ppps_packed_plan_env in {"1", "true", "on", "yes"}
    class_policy_env = _parse_direct_jk_class_policy_env(os.environ.get("ASUKA_DIRECT_JK_CLASS_POLICY", ""))
    if eri_block_pref_env:
        eri_block_pref = eri_block_pref_env in {"1", "true", "on", "yes"}
    else:
        eri_block_pref = False
    fused_supported = {
        "ssss", "psss", "dsss", "ppss", "psps", "ppps",
        "ddss", "ssdp", "psds", "psdp", "psdd", "ppds", "dsds", "dsdp",
        "ppdp", "ppdd", "dsdd", "dpdp", "dpdd", "dddd",
    }
    fused_enabled: set[str] = {"ssss", "psss", "psps"}
    fused_flag = str(os.environ.get("ASUKA_DIRECT_JK_FUSED", "") or "").strip().lower()
    if fused_flag in ("1", "true", "on", "yes"):
        fused_enabled = set(fused_supported)
    elif fused_flag in ("0", "false", "off", "no"):
        fused_enabled = set()
    fused_only = str(os.environ.get("ASUKA_DIRECT_JK_FUSED_ONLY", "") or "").strip()
    if fused_only:
        fused_enabled = {x.strip().lower() for x in fused_only.replace(" ", ",").split(",") if x.strip()}
    fused_disable = str(os.environ.get("ASUKA_DIRECT_JK_FUSED_DISABLE", "") or "").strip()
    if fused_disable:
        fused_enabled -= {x.strip().lower() for x in fused_disable.replace(" ", ",").split(",") if x.strip()}
    fused_enable = str(os.environ.get("ASUKA_DIRECT_JK_FUSED_ENABLE", "") or "").strip()
    if fused_enable:
        fused_enabled |= {x.strip().lower() for x in fused_enable.replace(" ", ",").split(",") if x.strip()}
    fused_enabled &= fused_supported
    class_policy = _default_direct_jk_class_policy(
        max_l=int(getattr(ctx, "max_l", 0)),
        large_workload=bool(int(ctx.ntasks) >= 1_000_000),
    )
    class_policy_env = _parse_direct_jk_class_policy_env(os.environ.get("ASUKA_DIRECT_JK_CLASS_POLICY", ""))
    if class_policy_env:
        class_policy.update(class_policy_env)

    # Multi-buffer J/K accumulation: distribute atomicAdd contention for fused kernels.
    n_bufs = _choose_n_bufs(nao) if fused_enabled else 1
    D_flat, J_flat, K_flat, J_bufs, K_bufs = _ensure_direct_jk_matrix_buffers(
        cp,
        ws,
        nao=int(nao),
        n_bufs=int(n_bufs),
        want_J=bool(want_J),
        want_K=bool(want_K),
    )
    if D_flat is None:
        D_flat = cp.ascontiguousarray(D_gpu.ravel())
        J_flat = cp.zeros((nao * nao,), dtype=cp.float64) if want_J else None
        K_flat = cp.zeros((nao * nao,), dtype=cp.float64) if want_K else None
    else:
        D_src = D_gpu.ravel()
        if not bool(getattr(D_src.flags, "c_contiguous", False)):
            D_src = cp.ascontiguousarray(D_src)
        D_flat[...] = D_src
        if J_flat is not None:
            J_flat.fill(0.0)
        if K_flat is not None:
            K_flat.fill(0.0)

    if n_bufs > 1:
        J_bufs = cp.zeros((n_bufs, nao * nao), dtype=cp.float64) if want_J else None
        K_bufs = cp.zeros((n_bufs, nao * nao), dtype=cp.float64) if want_K else None
    else:
        J_bufs = None
        K_bufs = None

    sp_pair_start_fused = ctx.dsp.sp_pair_start
    if int(getattr(sp_pair_start_fused, "shape", (0,))[0]) == int(getattr(ctx.dsp.sp_npair, "shape", (0,))[0]) + 1:
        sp_pair_start_fused = sp_pair_start_fused[:-1]

    if stats is not None:
        stats["direct_jk_ntasks"] = int(ctx.ntasks)
        stats.setdefault("resident_ntasks", 0)
        stats.setdefault("streamed_ntasks", 0)
        stats.setdefault("bytes_uploaded", 0)
        stats.setdefault("n_eval_calls", 0)
        stats.setdefault("n_contract_calls", 0)
        stats.setdefault("n_fused_calls", 0)
        stats.setdefault("fused_ntasks", 0)
        stats.setdefault("classes", {})
        if ppps_census_enabled:
            stats.setdefault("ppps_shape_census", {})
        if ppps_packed_plan_enabled:
            stats.setdefault("ppps_packed_plan", {})
    if profile is not None:
        profile.setdefault("direct_jk_classes", {})

    buf_slab: list[int | None] = [None, None]
    cur_buf = 0

    def _find_next_cpu(i: int) -> int | None:
        for j in range(int(i) + 1, int(len(ctx.slabs))):
            if not bool(ctx.slabs[j].gpu_resident):
                return int(j)
        return None

    def _select_engine(default_label: str, kernel_label: str, *, can_fuse: bool, warp_eri: bool, warp_contract: bool) -> tuple[str, bool, bool]:
        env_eng = _normalize_direct_jk_engine(
            class_policy_env.get(str(default_label).lower(), class_policy_env.get(str(kernel_label).lower(), ""))
        )
        if env_eng is not None:
            eng = str(env_eng)
        elif can_fuse:
            eng = "fused_jk"
        else:
            eng = str(class_policy.get(str(default_label).lower(), class_policy.get(str(kernel_label).lower(), "default")))
            eng = _normalize_direct_jk_engine(eng) or "default"
        if eng == "fused_jk":
            if can_fuse:
                return "fused_jk", False, False
            eng = "default"
        if eng == "staged_block":
            return "staged_block", False, False
        if eng == "staged_warp":
            return "staged_warp", True, True
        if eng == "staged_warp_eri":
            return "staged_warp_eri", True, False
        if eng == "staged_warp_contract":
            return "staged_warp_contract", False, True
        if can_fuse:
            return "fused_jk", False, False
        if bool(warp_eri) and bool(warp_contract):
            return "staged_warp", True, True
        if bool(warp_eri):
            return "staged_warp_eri", True, False
        if bool(warp_contract):
            return "staged_warp_contract", False, True
        return "staged_block", False, False

    def _has_explicit_class_policy(default_label: str, kernel_label: str) -> bool:
        env_eng = _normalize_direct_jk_engine(
            class_policy_env.get(str(default_label).lower(), class_policy_env.get(str(kernel_label).lower(), ""))
        )
        return env_eng is not None

    def _select_staged_eri_mode(default_label: str, kernel_label: str, *, use_warp_mode: bool) -> str:
        env_eng = _normalize_direct_jk_engine(
            class_policy_env.get(str(default_label).lower(), class_policy_env.get(str(kernel_label).lower(), ""))
        )
        if env_eng == "staged_block":
            return "block"
        if bool(use_warp_mode):
            if str(kernel_label).lower() == "ppps" and env_eng is None:
                return "auto"
            return "warp"
        return "block" if eri_block_pref else "auto"

    def _ensure_stats_row(bucket: dict[str, Any], cls: str, kernel_label: str, transpose: bool) -> dict[str, Any]:
        row = bucket.setdefault(
            cls,
            {
                "ntasks": 0,
                "chunks": 0,
                "path": "",
                "engine": "",
                "kernel_class": str(kernel_label),
                "transpose": int(bool(transpose)),
                "calls": 0,
            },
        )
        row["kernel_class"] = str(kernel_label)
        row["transpose"] = int(bool(transpose))
        return row

    def _enqueue_upload(slab_idx: int, buf_idx: int) -> None:
        assert ws is not None
        # Ensure we don't overwrite a buffer that compute is still reading.
        ws.copy_stream.wait_event(ws.compute_done[int(buf_idx)])

        slab_u = ctx.slabs[int(slab_idx)]
        nt = int(slab_u.ntasks)
        if nt <= 0:
            buf_slab[int(buf_idx)] = int(slab_idx)
            ws.upload_done[int(buf_idx)].record(ws.copy_stream)
            return
        if nt > int(ws.max_ntasks):
            raise RuntimeError(
                f"DirectJKWorkspace upload buffer too small: need ntasks={nt}, have {int(ws.max_ntasks)}"
            )
        ab_host = np.asarray(slab_u.ab_sorted, dtype=np.int32, order="C")
        cd_host = np.asarray(slab_u.cd_sorted, dtype=np.int32, order="C")
        dst_ab = ws.upload_ab[int(buf_idx)]
        dst_cd = ws.upload_cd[int(buf_idx)]
        stream_h2d = int(ws.copy_stream.ptr)
        cp.cuda.runtime.memcpyAsync(
            int(dst_ab.data.ptr),
            int(ab_host.ctypes.data),
            int(nt * 4),
            cp.cuda.runtime.memcpyHostToDevice,
            stream_h2d,
        )
        cp.cuda.runtime.memcpyAsync(
            int(dst_cd.data.ptr),
            int(cd_host.ctypes.data),
            int(nt * 4),
            cp.cuda.runtime.memcpyHostToDevice,
            stream_h2d,
        )
        ws.upload_done[int(buf_idx)].record(ws.copy_stream)
        buf_slab[int(buf_idx)] = int(slab_idx)
        if stats is not None:
            stats["bytes_uploaded"] = int(stats.get("bytes_uploaded", 0)) + int(nt * 8)

    for slab_i, slab in enumerate(ctx.slabs):
        slab_i = int(slab_i)

        # Prefetch the next CPU slab during GPU-resident slab compute when possible.
        if ws is not None and bool(slab.gpu_resident):
            nxt = _find_next_cpu(slab_i)
            if nxt is not None and nxt not in buf_slab:
                free_buf = 0 if buf_slab[0] is None else (1 if buf_slab[1] is None else None)
                if free_buf is not None:
                    _enqueue_upload(int(nxt), int(free_buf))

        used_buf: int | None = None
        if slab.gpu_resident:
            ab_dev = slab.ab_sorted
            cd_dev = slab.cd_sorted
            if stats is not None:
                stats["resident_ntasks"] = int(stats.get("resident_ntasks", 0)) + int(slab.ntasks)
        else:
            if stats is not None:
                stats["streamed_ntasks"] = int(stats.get("streamed_ntasks", 0)) + int(slab.ntasks)
            if ws is None:
                # Fallback: synchronous upload.
                ab_dev = cp.ascontiguousarray(cp.asarray(slab.ab_sorted, dtype=cp.int32))
                cd_dev = cp.ascontiguousarray(cp.asarray(slab.cd_sorted, dtype=cp.int32))
                if stats is not None:
                    stats["bytes_uploaded"] = int(stats.get("bytes_uploaded", 0)) + int(slab.ntasks * 8)
            else:
                # Ensure current slab is uploaded into one of the two buffers.
                if slab_i in buf_slab:
                    cur_buf = int(buf_slab.index(slab_i))
                else:
                    if buf_slab[int(cur_buf)] is not None:
                        cur_buf = 1 - int(cur_buf)
                    _enqueue_upload(int(slab_i), int(cur_buf))

                # Prefetch the next CPU slab into the other buffer (copy stream).
                nxt = _find_next_cpu(slab_i)
                other = 1 - int(cur_buf)
                if nxt is not None and nxt not in buf_slab and buf_slab[int(other)] is None:
                    _enqueue_upload(int(nxt), int(other))

                # Wait for H2D completion on the compute stream.
                compute_stream.wait_event(ws.upload_done[int(cur_buf)])
                nt = int(slab.ntasks)
                ab_dev = ws.upload_ab[int(cur_buf)][:nt]
                cd_dev = ws.upload_cd[int(cur_buf)][:nt]
                used_buf = int(cur_buf)

        slab_plan = ctx.plans[slab_i]
        for g, gp in enumerate(slab_plan):
            orig_cid = int(gp.orig_cid)
            cls = str(gp.class_label)
            klabel = str(gp.kernel_label).lower()
            j0, j1 = int(gp.j0), int(gp.j1)
            if j1 <= j0:
                continue

            kernel_cid = int(gp.kernel_cid)
            transpose = bool(gp.transpose)
            hot_s1 = bool(gp.hot_s1)
            nA = int(gp.nA)
            nB = int(gp.nB)
            nC = int(gp.nC)
            nD = int(gp.nD)
            chunk_ntasks = int(gp.chunk_ntasks)
            default_warp_mode = bool(gp.use_warp_eri)
            default_warp_contract = bool(gp.use_warp_contract)

            kernel_spAB_full = cd_dev[j0:j1] if transpose else ab_dev[j0:j1]
            kernel_spCD_full = ab_dev[j0:j1] if transpose else cd_dev[j0:j1]

            class_ntasks = j1 - j0

            # Density prescreening (CSAM): filter tasks where
            # Q[ab] * Q[cd] * max(D_sp[ab], D_sp[cd]) < eps_density
            if D_sp_dev is not None and Q_sp is not None and class_ntasks > 0:
                _q_ab = Q_sp[kernel_spAB_full]
                _q_cd = Q_sp[kernel_spCD_full]
                _d_ab = D_sp_dev[kernel_spAB_full]
                _d_cd = D_sp_dev[kernel_spCD_full]
                _screen_val = _q_ab * _q_cd * cp.maximum(_d_ab, _d_cd)
                _mask = _screen_val >= eps_density
                n_survive = int(_mask.sum())
                if n_survive < class_ntasks:
                    if n_survive == 0:
                        continue
                    _idx = cp.nonzero(_mask)[0]
                    kernel_spAB_full = kernel_spAB_full[_idx]
                    kernel_spCD_full = kernel_spCD_full[_idx]
                    class_ntasks = n_survive

            if ppps_census_enabled and stats is not None and klabel == "ppps" and class_ntasks > 0:
                _nAB = cp.asnumpy(ctx.dsp.sp_npair[kernel_spAB_full]).astype(np.int64, copy=False)
                _nCD = cp.asnumpy(ctx.dsp.sp_npair[kernel_spCD_full]).astype(np.int64, copy=False)
                _accumulate_ppps_shape_census(
                    stats.setdefault("ppps_shape_census", {}),
                    cls=cls,
                    transpose=transpose,
                    nAB=_nAB,
                    nCD=_nCD,
                )
            if ppps_packed_plan_enabled and stats is not None and klabel == "ppps" and class_ntasks > 0:
                _sp_npair = cp.asnumpy(ctx.dsp.sp_npair).astype(np.int32, copy=False)
                _plan = _build_ppps_packed_slab_plan(
                    task_spAB=cp.asnumpy(kernel_spAB_full).astype(np.int32, copy=False),
                    task_spCD=cp.asnumpy(kernel_spCD_full).astype(np.int32, copy=False),
                    sp_npair=_sp_npair,
                    transpose=transpose,
                )
                _accumulate_ppps_packed_plan_stats(stats.setdefault("ppps_packed_plan", {}), _plan)

            fused_fn = None
            if klabel in fused_enabled:
                if klabel == "ssss":
                    fused_fn = getattr(_ext, "fused_jk_ssss_inplace_device", None)
                else:
                    fused_fn = getattr(_ext, f"fused_jk_{klabel}_inplace_device", None)
            use_fused = fused_fn is not None
            engine, use_warp_mode, use_warp_contract = _select_engine(
                cls,
                klabel,
                can_fuse=bool(use_fused),
                warp_eri=bool(default_warp_mode),
                warp_contract=bool(default_warp_contract),
            )
            use_fused = bool(engine == "fused_jk" and fused_fn is not None)
            if stats is not None:
                row = _ensure_stats_row(stats.setdefault("classes", {}), cls, klabel, transpose)
                row["ntasks"] = int(row.get("ntasks", 0)) + int(class_ntasks)
                row["chunks"] = int(row.get("chunks", 0)) + (1 if use_fused else int((class_ntasks + chunk_ntasks - 1) // chunk_ntasks))
                path = str(engine)
                if (not use_fused) and hot_s1 and hot_s1_contract_enabled:
                    path = f"{path}_hot_s1"
                prev = str(row.get("path", "") or "")
                row["path"] = path if (not prev or prev == path) else "mixed"
                prev_eng = str(row.get("engine", "") or "")
                row["engine"] = str(engine) if (not prev_eng or prev_eng == str(engine)) else "mixed"
                row["calls"] = int(row.get("calls", 0)) + (1 if use_fused else int((class_ntasks + chunk_ntasks - 1) // chunk_ntasks))
            prof_row = None
            if profile is not None:
                prof_row = _ensure_stats_row(profile.setdefault("direct_jk_classes", {}), cls, klabel, transpose)
                prof_row["ntasks"] = int(prof_row.get("ntasks", 0)) + int(class_ntasks)
                prof_row["chunks"] = int(prof_row.get("chunks", 0)) + (1 if use_fused else int((class_ntasks + chunk_ntasks - 1) // chunk_ntasks))
                prof_row["engine"] = str(engine)
                prof_row.setdefault("kernel_ms", 0.0)
                prof_row.setdefault("contract_ms", 0.0)
                prof_row.setdefault("fused_ms", 0.0)
                prof_row["calls"] = int(prof_row.get("calls", 0)) + (1 if use_fused else int((class_ntasks + chunk_ntasks - 1) // chunk_ntasks))

            if use_fused and fused_fn is not None:
                if profile is not None:
                    _tc0 = cp.cuda.Event()
                    _tc1 = cp.cuda.Event()
                    _tc0.record()
                _J_target = J_bufs.ravel() if J_bufs is not None else J_flat
                _K_target = K_bufs.ravel() if K_bufs is not None else K_flat
                if klabel == "ssss":
                    fused_fn(
                        kernel_spAB_full,
                        kernel_spCD_full,
                        ctx.dsp.sp_pair_start,
                        ctx.dsp.sp_npair,
                        ctx.pair_tables.pair_eta,
                        ctx.pair_tables.pair_Px,
                        ctx.pair_tables.pair_Py,
                        ctx.pair_tables.pair_Pz,
                        ctx.pair_tables.pair_cK,
                        ctx.sp_A_dev,
                        ctx.sp_B_dev,
                        ctx.shell_ao_start_dev,
                        int(nao),
                        D_flat,
                        _J_target,
                        _K_target,
                        int(eval_threads),
                        int(stream_ptr),
                        False,
                        False,
                        n_bufs=int(n_bufs),
                    )
                else:
                    fused_fn(
                        kernel_spAB_full,
                        kernel_spCD_full,
                        ctx.dsp.sp_A,
                        ctx.dsp.sp_B,
                        sp_pair_start_fused,
                        ctx.dsp.sp_npair,
                        ctx.dbasis.shell_cx,
                        ctx.dbasis.shell_cy,
                        ctx.dbasis.shell_cz,
                        ctx.pair_tables.pair_eta,
                        ctx.pair_tables.pair_Px,
                        ctx.pair_tables.pair_Py,
                        ctx.pair_tables.pair_Pz,
                        ctx.pair_tables.pair_cK,
                        ctx.shell_ao_start_dev,
                        int(nao),
                        D_flat,
                        _J_target,
                        _K_target,
                        int(eval_threads),
                        int(stream_ptr),
                        False,
                        n_bufs=int(n_bufs),
                    )
                n_kernel_calls += 1
                if stats is not None:
                    stats["n_fused_calls"] = int(stats.get("n_fused_calls", 0)) + 1
                    stats["n_eval_calls"] = int(stats.get("n_eval_calls", 0)) + 1
                    stats["fused_ntasks"] = int(stats.get("fused_ntasks", 0)) + int(class_ntasks)
                if profile is not None:
                    _tc1.record()
                    _tc1.synchronize()
                    _dt = float(cp.cuda.get_elapsed_time(_tc0, _tc1))
                    profile["contract_ms"] = float(profile.get("contract_ms", 0.0)) + _dt
                    profile["fused_jk_ms"] = float(profile.get("fused_jk_ms", 0.0)) + _dt
                    if prof_row is not None:
                        prof_row["fused_ms"] = float(prof_row.get("fused_ms", 0.0)) + float(_dt)
                continue

            for c0 in range(0, class_ntasks, chunk_ntasks):
                c1 = min(class_ntasks, c0 + chunk_ntasks)
                if c1 <= c0:
                    continue
                eri_mode = _select_staged_eri_mode(cls, klabel, use_warp_mode=use_warp_mode)
                kernel_ms_before = float(profile.get("kernel_ms", 0.0)) if profile is not None else 0.0
                sub_batch = KernelBatch(
                    task_idx=np.empty(0, dtype=np.int32),  # unused by run_kernel_batch_spd
                    kernel_tasks=TaskList(
                        task_spAB=kernel_spAB_full[c0:c1],
                        task_spCD=kernel_spCD_full[c0:c1],
                    ),
                    kernel_class_id=np.int32(kernel_cid),
                    transpose=transpose,
                )
                tiles = run_kernel_batch_spd(
                    sub_batch,
                    dbasis=ctx.dbasis,
                    dsp=ctx.dsp,
                    pt=ctx.pair_tables,
                    stream=None,
                    threads=eval_threads,
                    mode=eri_mode,
                    profile=profile,
                    skip_transpose=True,
                )
                n_kernel_calls += 1
                if stats is not None:
                    stats["n_eval_calls"] = int(stats.get("n_eval_calls", 0)) + 1
                if profile is not None and prof_row is not None:
                    kernel_dt = float(profile.get("kernel_ms", 0.0)) - float(kernel_ms_before)
                    prof_row["kernel_ms"] = float(prof_row.get("kernel_ms", 0.0)) + float(kernel_dt)

                if profile is not None:
                    _tc0 = cp.cuda.Event()
                    _tc1 = cp.cuda.Event()
                    _tc0.record()

                contract_fn = (
                    _ext.contract_jk_tiles_ordered_warp_inplace_device
                    if use_warp_contract
                    else _ext.contract_jk_tiles_ordered_inplace_device
                )

                # When transpose=True, the kernel produced tiles in
                # (nCD_orig, nAB_orig) layout.  Instead of transposing +
                # copying tiles, swap nA/nB/nC/nD and shell-pair arrays
                # passed to the contraction kernel.
                if transpose:
                    contract_fn(
                        cd_dev[j0 + c0: j0 + c1],
                        ab_dev[j0 + c0: j0 + c1],
                        ctx.sp_A_dev,
                        ctx.sp_B_dev,
                        ctx.shell_ao_start_dev,
                        int(nao),
                        int(nC),
                        int(nD),
                        int(nA),
                        int(nB),
                        tiles.ravel(),
                        D_flat,
                        J_flat,
                        K_flat,
                        int(contract_threads),
                        int(stream_ptr),
                        False,
                    )
                else:
                    contract_fn(
                        ab_dev[j0 + c0: j0 + c1],
                        cd_dev[j0 + c0: j0 + c1],
                        ctx.sp_A_dev,
                        ctx.sp_B_dev,
                        ctx.shell_ao_start_dev,
                        int(nao),
                        int(nA),
                        int(nB),
                        int(nC),
                        int(nD),
                        tiles.ravel(),
                        D_flat,
                        J_flat,
                        K_flat,
                        int(contract_threads),
                        int(stream_ptr),
                        False,
                    )
                if stats is not None:
                    stats["n_contract_calls"] = int(stats.get("n_contract_calls", 0)) + 1

                if profile is not None:
                    _tc1.record()
                    _tc1.synchronize()
                    contract_dt = float(cp.cuda.get_elapsed_time(_tc0, _tc1))
                    profile["contract_ms"] = float(profile.get("contract_ms", 0.0)) + float(contract_dt)
                    if prof_row is not None:
                        prof_row["contract_ms"] = float(prof_row.get("contract_ms", 0.0)) + float(contract_dt)

        if used_buf is not None and ws is not None:
            # Prevent buffer reuse until all compute that reads it is complete.
            ws.compute_done[int(used_buf)].record(compute_stream)
            if slab_i in buf_slab:
                bi = int(buf_slab.index(slab_i))
                buf_slab[bi] = None

    # Reduce multi-buffer J/K contributions.
    if J_bufs is not None and J_flat is not None:
        J_flat += J_bufs.sum(axis=0)
    if K_bufs is not None and K_flat is not None:
        K_flat += K_bufs.sum(axis=0)

    J = None
    K = None
    if J_flat is not None:
        J = J_flat.reshape((nao, nao))
        J = 0.5 * (J + J.T)
    if K_flat is not None:
        K = K_flat.reshape((nao, nao))
        K = 0.5 * (K + K.T)

    if profile is not None:
        profile["direct_jk_t_s"] = float(time.perf_counter() - t0)
        profile["direct_jk_kernel_calls"] = int(n_kernel_calls)
        profile["direct_jk_ntasks"] = int(ctx.ntasks)

    return J, K


def direct_fock_rhf(
    ctx: DirectJKContext,
    D,
    hcore,
    *,
    profile: dict | None = None,
    stats: dict | None = None,
):
    """Build the RHF Fock matrix directly: F = h + J(D) - 0.5 * K(D).

    This is an SCF-specific convenience that avoids allocating materialized
    J and K outputs.
    """

    import cupy as cp  # noqa: PLC0415

    _ext = cueri_kernels.require_ext()
    from asuka.cueri.eri_dispatch import KernelBatch, run_kernel_batch_spd  # noqa: PLC0415
    from asuka.cueri.tasks import TaskList  # noqa: PLC0415
    import os  # noqa: PLC0415

    nao = ctx.nao
    base_threads = int(ctx.threads)
    eval_threads = int(getattr(ctx, "eval_threads", base_threads))
    contract_threads = int(getattr(ctx, "contract_threads", base_threads))

    D_gpu = cp.asarray(D, dtype=cp.float64)
    if D_gpu.ndim != 2 or D_gpu.shape != (nao, nao):
        raise ValueError(f"D must be ({nao}, {nao}), got {tuple(D_gpu.shape)}")
    D_flat = cp.ascontiguousarray(D_gpu.ravel())

    h_gpu = cp.asarray(hcore, dtype=cp.float64)
    if h_gpu.ndim != 2 or h_gpu.shape != (nao, nao):
        raise ValueError(f"hcore must be ({nao}, {nao}), got {tuple(h_gpu.shape)}")
    F_flat = cp.ascontiguousarray(h_gpu.ravel()).copy()

    # Density prescreening (CSAM) for direct_fock_rhf
    eps_density_env = os.environ.get("ASUKA_DIRECT_JK_EPS_DENSITY", "")
    eps_density = float(eps_density_env) if eps_density_env.strip() else max(ctx.eps_schwarz, 1e-10)
    Q_sp = getattr(ctx, "Q_sp", None)
    shell_l_dev = getattr(ctx, "shell_l_dev", None)
    D_sp_dev = None
    if Q_sp is not None and shell_l_dev is not None and eps_density > 0:
        D_sp_dev = _compute_D_sp_max(
            D_gpu, ctx.sp_A_dev, ctx.sp_B_dev, ctx.shell_ao_start_dev,
            shell_l_dev, ctx.nsp, cp,
        )

    compute_stream = cp.cuda.get_current_stream()
    stream_ptr = int(compute_stream.ptr)
    ws = _get_direct_jk_workspace(cp, ctx)
    n_kernel_calls = 0
    t0 = time.perf_counter()
    hot_s1_env = str(os.environ.get("ASUKA_DENSE_CONTRACT_HOT_S1", "") or "").strip().lower()
    hot_s1_contract_enabled = hot_s1_env not in {"0", "false", "off", "no"}
    eri_block_pref_env = str(os.environ.get("ASUKA_DIRECT_JK_ERI_BLOCK_MODE", "") or "").strip().lower()
    class_policy_env = _parse_direct_jk_class_policy_env(os.environ.get("ASUKA_DIRECT_JK_CLASS_POLICY", ""))
    if eri_block_pref_env:
        eri_block_pref = eri_block_pref_env in {"1", "true", "on", "yes"}
    else:
        eri_block_pref = False

    if stats is not None:
        stats["direct_jk_ntasks"] = int(ctx.ntasks)
        stats.setdefault("resident_ntasks", 0)
        stats.setdefault("streamed_ntasks", 0)
        stats.setdefault("bytes_uploaded", 0)
        stats.setdefault("n_eval_calls", 0)
        stats.setdefault("n_contract_calls", 0)
        stats.setdefault("n_fused_calls", 0)
        stats.setdefault("fused_ntasks", 0)
        stats.setdefault("classes", {})

    buf_slab: list[int | None] = [None, None]
    cur_buf = 0

    def _find_next_cpu(i: int) -> int | None:
        for j in range(int(i) + 1, int(len(ctx.slabs))):
            if not bool(ctx.slabs[j].gpu_resident):
                return int(j)
        return None

    def _am_label(l: int) -> str:
        am = "spdfghijklm"
        li = int(l)
        if 0 <= li < len(am):
            return am[li]
        return f"l{li}"

    def _class_label(cid: int) -> str:
        x = int(cid) & 0xFFFFFFFF
        la = x & 0xFF
        lb = (x >> 8) & 0xFF
        lc = (x >> 16) & 0xFF
        ld = (x >> 24) & 0xFF
        return f"{_am_label(la)}{_am_label(lb)}{_am_label(lc)}{_am_label(ld)}"

    def _has_explicit_class_policy(default_label: str, kernel_label: str) -> bool:
        env_eng = _normalize_direct_jk_engine(
            class_policy_env.get(str(default_label).lower(), class_policy_env.get(str(kernel_label).lower(), ""))
        )
        return env_eng is not None

    def _select_staged_eri_mode(default_label: str, kernel_label: str, *, use_warp_mode: bool) -> str:
        env_eng = _normalize_direct_jk_engine(
            class_policy_env.get(str(default_label).lower(), class_policy_env.get(str(kernel_label).lower(), ""))
        )
        if env_eng == "staged_block":
            return "block"
        if bool(use_warp_mode):
            if str(kernel_label).lower() == "ppps" and env_eng is None:
                return "auto"
            return "warp"
        return "block" if eri_block_pref else "auto"

    sp_pair_start_fused = ctx.dsp.sp_pair_start
    if int(sp_pair_start_fused.shape[0]) == int(ctx.dsp.sp_npair.shape[0]) + 1:
        # Some contexts store a sentinel end entry (nsp+1); fused CUDA APIs expect nsp.
        sp_pair_start_fused = sp_pair_start_fused[:-1]

    # All fused kernels use warp-per-task pattern.
    # ssss: scalar ERI + direct Fock (cueri_cuda_kernels.cu)
    # psss/dsss/ppss/psps/ppps: explicit Boys derivatives (step2.cu)
    # ddss/ssdp/psds/psdp/psdd/ppds/dsds/dsdp: Rys quadrature, converted
    #   from block-per-task to warp-per-task (wave1/wave2 generated files)
    fused_supported = {
        "ssss", "psss", "dsss", "ppss", "psps", "ppps",
        "ddss", "ssdp", "psds", "psdp", "ppds", "dsds", "dsdp", "psdd",
        "ppdp", "ppdd", "dsdd", "dpdp", "dpdd", "dddd",
    }
    # Safety-first default: keep fused direct-Fock kernels opt-in until strict
    # SCF parity coverage is in place for every fused class.
    fused_enabled: set[str] = set()
    fused_flag = str(os.environ.get("ASUKA_DIRECT_FOCK_FUSED", "") or "").strip().lower()
    if fused_flag in ("1", "true", "on", "yes"):
        fused_enabled = set(fused_supported)
    elif fused_flag in ("0", "false", "off", "no"):
        fused_enabled = set()
    fused_only = str(os.environ.get("ASUKA_DIRECT_FOCK_FUSED_ONLY", "") or "").strip()
    if fused_only:
        fused_enabled = {x.strip().lower() for x in fused_only.replace(" ", ",").split(",") if x.strip()}
    fused_disable = str(os.environ.get("ASUKA_DIRECT_FOCK_FUSED_DISABLE", "") or "").strip()
    if fused_disable:
        fused_enabled -= {x.strip().lower() for x in fused_disable.replace(" ", ",").split(",") if x.strip()}
    fused_enable = str(os.environ.get("ASUKA_DIRECT_FOCK_FUSED_ENABLE", "") or "").strip()
    if fused_enable:
        fused_enabled |= {x.strip().lower() for x in fused_enable.replace(" ", ",").split(",") if x.strip()}
    fused_enabled &= fused_supported

    # Multi-buffer Fock accumulation: distribute atomicAdd contention for fused kernels.
    n_bufs = _choose_n_bufs(nao) if fused_enabled else 1
    if n_bufs > 1:
        F_bufs = cp.zeros((n_bufs, nao * nao), dtype=cp.float64)
    else:
        F_bufs = None

    def _enqueue_upload(slab_idx: int, buf_idx: int) -> None:
        assert ws is not None
        # Ensure we don't overwrite a buffer that compute is still reading.
        ws.copy_stream.wait_event(ws.compute_done[int(buf_idx)])

        slab_u = ctx.slabs[int(slab_idx)]
        nt = int(slab_u.ntasks)
        if nt <= 0:
            buf_slab[int(buf_idx)] = int(slab_idx)
            ws.upload_done[int(buf_idx)].record(ws.copy_stream)
            return
        if nt > int(ws.max_ntasks):
            raise RuntimeError(
                f"DirectJKWorkspace upload buffer too small: need ntasks={nt}, have {int(ws.max_ntasks)}"
            )
        ab_host = np.asarray(slab_u.ab_sorted, dtype=np.int32, order="C")
        cd_host = np.asarray(slab_u.cd_sorted, dtype=np.int32, order="C")
        dst_ab = ws.upload_ab[int(buf_idx)]
        dst_cd = ws.upload_cd[int(buf_idx)]
        stream_h2d = int(ws.copy_stream.ptr)
        cp.cuda.runtime.memcpyAsync(
            int(dst_ab.data.ptr),
            int(ab_host.ctypes.data),
            int(nt * 4),
            cp.cuda.runtime.memcpyHostToDevice,
            stream_h2d,
        )
        cp.cuda.runtime.memcpyAsync(
            int(dst_cd.data.ptr),
            int(cd_host.ctypes.data),
            int(nt * 4),
            cp.cuda.runtime.memcpyHostToDevice,
            stream_h2d,
        )
        ws.upload_done[int(buf_idx)].record(ws.copy_stream)
        buf_slab[int(buf_idx)] = int(slab_idx)
        if stats is not None:
            stats["bytes_uploaded"] = int(stats.get("bytes_uploaded", 0)) + int(nt * 8)

    for slab_i, slab in enumerate(ctx.slabs):
        slab_i = int(slab_i)

        # Prefetch the next CPU slab during GPU-resident slab compute when possible.
        if ws is not None and bool(slab.gpu_resident):
            nxt = _find_next_cpu(slab_i)
            if nxt is not None and nxt not in buf_slab:
                free_buf = 0 if buf_slab[0] is None else (1 if buf_slab[1] is None else None)
                if free_buf is not None:
                    _enqueue_upload(int(nxt), int(free_buf))

        used_buf: int | None = None
        if slab.gpu_resident:
            ab_dev = slab.ab_sorted
            cd_dev = slab.cd_sorted
            if stats is not None:
                stats["resident_ntasks"] = int(stats.get("resident_ntasks", 0)) + int(slab.ntasks)
        else:
            if stats is not None:
                stats["streamed_ntasks"] = int(stats.get("streamed_ntasks", 0)) + int(slab.ntasks)
            if ws is None:
                # Fallback: synchronous upload.
                ab_dev = cp.ascontiguousarray(cp.asarray(slab.ab_sorted, dtype=cp.int32))
                cd_dev = cp.ascontiguousarray(cp.asarray(slab.cd_sorted, dtype=cp.int32))
                if stats is not None:
                    stats["bytes_uploaded"] = int(stats.get("bytes_uploaded", 0)) + int(slab.ntasks * 8)
            else:
                # Ensure current slab is uploaded into one of the two buffers.
                if slab_i in buf_slab:
                    cur_buf = int(buf_slab.index(slab_i))
                else:
                    if buf_slab[int(cur_buf)] is not None:
                        cur_buf = 1 - int(cur_buf)
                    _enqueue_upload(int(slab_i), int(cur_buf))

                # Prefetch the next CPU slab into the other buffer (copy stream).
                nxt = _find_next_cpu(slab_i)
                other = 1 - int(cur_buf)
                if nxt is not None and nxt not in buf_slab and buf_slab[int(other)] is None:
                    _enqueue_upload(int(nxt), int(other))

                # Wait for H2D completion on the compute stream.
                compute_stream.wait_event(ws.upload_done[int(cur_buf)])
                nt = int(slab.ntasks)
                ab_dev = ws.upload_ab[int(cur_buf)][:nt]
                cd_dev = ws.upload_cd[int(cur_buf)][:nt]
                used_buf = int(cur_buf)

        # --- Bulk density prescreening for direct_fock_rhf ---
        slab_offsets_adj = None
        slab_ntasks_eff = int(slab.ntasks)
        if D_sp_dev is not None and Q_sp is not None and slab_ntasks_eff > 0:
            _ab_slab = ab_dev[:slab_ntasks_eff]
            _cd_slab = cd_dev[:slab_ntasks_eff]
            _screen_all = Q_sp[_ab_slab] * Q_sp[_cd_slab] * cp.maximum(D_sp_dev[_ab_slab], D_sp_dev[_cd_slab])
            _mask_all = _screen_all >= eps_density
            n_survive_total = int(_mask_all.sum())
            if n_survive_total < slab_ntasks_eff:
                if n_survive_total == 0:
                    if used_buf is not None and ws is not None:
                        ws.compute_done[int(used_buf)].record(compute_stream)
                        buf_slab[int(used_buf)] = None
                    continue
                _idx_all = cp.nonzero(_mask_all)[0]
                ab_dev = ab_dev[_idx_all]
                cd_dev = cd_dev[_idx_all]
                # Recompute per-group offsets on GPU (only transfer ~N_groups values).
                old_offsets = slab.offsets
                _cumsum_dev = cp.cumsum(_mask_all)
                _odev = cp.asarray(old_offsets, dtype=cp.int64)
                # cumsum[end-1] gives count of surviving tasks up to group boundary.
                # Clamp indices to 0 to handle empty leading groups (offset==0).
                _boundary_idx = cp.maximum(_odev[1:] - 1, 0)  # (n_groups-1,)
                _boundary_vals = _cumsum_dev[_boundary_idx]
                # Where original offset was 0, the new offset must also be 0.
                _boundary_vals[_odev[1:] == 0] = 0
                slab_offsets_adj = np.empty(len(old_offsets), dtype=np.int64)
                slab_offsets_adj[0] = 0
                slab_offsets_adj[1:] = cp.asnumpy(_boundary_vals)
                del _cumsum_dev, _odev, _boundary_idx, _boundary_vals

        slab_plan = ctx.plans[slab_i]
        for gpi, gp in enumerate(slab_plan):
            orig_cid = int(gp.orig_cid)
            if slab_offsets_adj is not None:
                j0 = int(slab_offsets_adj[gpi])
                j1 = int(slab_offsets_adj[gpi + 1])
            else:
                j0, j1 = int(gp.j0), int(gp.j1)
            if j1 <= j0:
                continue

            kernel_cid = int(gp.kernel_cid)
            transpose = bool(gp.transpose)
            nA = int(gp.nA)
            nB = int(gp.nB)
            nC = int(gp.nC)
            nD = int(gp.nD)
            chunk_ntasks = int(gp.chunk_ntasks)
            use_warp_mode = bool(gp.use_warp_eri)
            use_warp_contract = bool(gp.use_warp_contract)

            kernel_spAB_full = cd_dev[j0:j1] if transpose else ab_dev[j0:j1]
            kernel_spCD_full = ab_dev[j0:j1] if transpose else cd_dev[j0:j1]

            class_ntasks = j1 - j0

            klabel = _class_label(kernel_cid).lower()
            fused_fn = None
            use_fused = False
            if klabel in fused_enabled:
                fused_fn = getattr(_ext, f"fused_fock_{klabel}_inplace_device", None)
                use_fused = fused_fn is not None

            if stats is not None:
                cls = _class_label(orig_cid)
                row = stats.setdefault("classes", {}).setdefault(cls, {"ntasks": 0, "chunks": 0, "path": ""})
                row["ntasks"] = int(row.get("ntasks", 0)) + int(class_ntasks)
                if use_fused:
                    row["chunks"] = int(row.get("chunks", 0)) + 1
                else:
                    row["chunks"] = int(row.get("chunks", 0)) + int((class_ntasks + chunk_ntasks - 1) // chunk_ntasks)
                hot_s1 = bool(
                    (int(nB) == 1 and int(nD) == 1)
                    and (int(nA) in (1, 3, 6))
                    and (int(nC) in (1, 3, 6))
                )
                path = "fock_fused" if use_fused else ("fock_staged_hot_s1" if (hot_s1 and hot_s1_contract_enabled) else "fock_staged")
                prev = str(row.get("path", "") or "")
                row["path"] = path if (not prev or prev == path) else "mixed"

            if use_fused and fused_fn is not None:
                # One fused kernel per class group (no tile allocation / contraction launch).
                fused_fn(
                    kernel_spAB_full,
                    kernel_spCD_full,
                    ctx.dsp.sp_A,
                    ctx.dsp.sp_B,
                    sp_pair_start_fused,
                    ctx.dsp.sp_npair,
                    ctx.dbasis.shell_cx,
                    ctx.dbasis.shell_cy,
                    ctx.dbasis.shell_cz,
                    ctx.pair_tables.pair_eta,
                    ctx.pair_tables.pair_Px,
                    ctx.pair_tables.pair_Py,
                    ctx.pair_tables.pair_Pz,
                    ctx.pair_tables.pair_cK,
                    ctx.shell_ao_start_dev,
                    int(nao),
                    D_flat,
                    F_bufs.ravel() if F_bufs is not None else F_flat,
                    int(eval_threads),
                    int(stream_ptr),
                    False,
                    n_bufs=int(n_bufs),
                )
                n_kernel_calls += 1
                if stats is not None:
                    stats["n_fused_calls"] = int(stats.get("n_fused_calls", 0)) + 1
                    stats["n_eval_calls"] = int(stats.get("n_eval_calls", 0)) + 1
                    stats["fused_ntasks"] = int(stats.get("fused_ntasks", 0)) + int(class_ntasks)
                continue

            for c0 in range(0, class_ntasks, chunk_ntasks):
                c1 = min(class_ntasks, c0 + chunk_ntasks)
                if c1 <= c0:
                    continue

                sub_batch = KernelBatch(
                    task_idx=np.empty(0, dtype=np.int32),  # unused by run_kernel_batch_spd
                    kernel_tasks=TaskList(
                        task_spAB=kernel_spAB_full[c0:c1],
                        task_spCD=kernel_spCD_full[c0:c1],
                    ),
                    kernel_class_id=np.int32(kernel_cid),
                    transpose=transpose,
                )

                cls = _class_label(orig_cid)
                klabel = _class_label(kernel_cid).lower()
                eri_mode = _select_staged_eri_mode(cls, klabel, use_warp_mode=use_warp_mode)
                tiles = run_kernel_batch_spd(
                    sub_batch,
                    dbasis=ctx.dbasis,
                    dsp=ctx.dsp,
                    pt=ctx.pair_tables,
                    stream=None,
                    threads=eval_threads,
                    mode=eri_mode,
                    profile=profile,
                    skip_transpose=True,
                )
                n_kernel_calls += 1
                if stats is not None:
                    stats["n_eval_calls"] = int(stats.get("n_eval_calls", 0)) + 1

                if profile is not None:
                    _tc0 = cp.cuda.Event()
                    _tc1 = cp.cuda.Event()
                    _tc0.record()

                contract_fn = (
                    _ext.contract_fock_tiles_ordered_warp_inplace_device
                    if use_warp_contract
                    else _ext.contract_fock_tiles_ordered_inplace_device
                )

                if transpose:
                    contract_fn(
                        cd_dev[j0 + c0: j0 + c1],
                        ab_dev[j0 + c0: j0 + c1],
                        ctx.sp_A_dev,
                        ctx.sp_B_dev,
                        ctx.shell_ao_start_dev,
                        int(nao),
                        int(nC),
                        int(nD),
                        int(nA),
                        int(nB),
                        tiles.ravel(),
                        D_flat,
                        F_flat,
                        int(contract_threads),
                        int(stream_ptr),
                        False,
                    )
                else:
                    contract_fn(
                        ab_dev[j0 + c0: j0 + c1],
                        cd_dev[j0 + c0: j0 + c1],
                        ctx.sp_A_dev,
                        ctx.sp_B_dev,
                        ctx.shell_ao_start_dev,
                        int(nao),
                        int(nA),
                        int(nB),
                        int(nC),
                        int(nD),
                        tiles.ravel(),
                        D_flat,
                        F_flat,
                        int(contract_threads),
                        int(stream_ptr),
                        False,
                    )
                if stats is not None:
                    stats["n_contract_calls"] = int(stats.get("n_contract_calls", 0)) + 1

                if profile is not None:
                    _tc1.record()
                    _tc1.synchronize()
                    profile["contract_ms"] = float(profile.get("contract_ms", 0.0)) + float(
                        cp.cuda.get_elapsed_time(_tc0, _tc1)
                    )

        if used_buf is not None and ws is not None:
            # Prevent buffer reuse until all compute that reads it is complete.
            ws.compute_done[int(used_buf)].record(compute_stream)
            if slab_i in buf_slab:
                bi = int(buf_slab.index(slab_i))
                buf_slab[bi] = None

    # Reduce multi-buffer Fock contributions into F_flat.
    if F_bufs is not None:
        F_flat += F_bufs.sum(axis=0)

    F = F_flat.reshape((nao, nao))
    F = 0.5 * (F + F.T)

    if profile is not None:
        profile["direct_fock_t_s"] = float(time.perf_counter() - t0)
        profile["direct_fock_kernel_calls"] = int(n_kernel_calls)
        profile["direct_fock_ntasks"] = int(ctx.ntasks)

    return F


def direct_JK_multi(
    ctx: DirectJKContext,
    Da,
    Db,
    *,
    want_J: bool = True,
    want_K: bool = True,
    profile: dict | None = None,
    stats: dict | None = None,
):
    """Build (Ja, Ka, Jb, Kb) evaluating each ERI tile once.

    For UHF/ROHF this is ~2x faster than calling ``direct_JK`` twice because
    the ERI evaluation (94% of runtime) is shared.

    Parameters
    ----------
    profile
        Optional dict for timing statistics.  **Note**: profile mode inserts
        ``event.synchronize()`` after each contraction kernel call, serializing
        execution.  Reported timings are not representative of production
        throughput.

    Returns
    -------
    (Ja, Ka, Jb, Kb) : tuple
        Coulomb/exchange matrices for each density, each ``(nao, nao)`` or None.
    """

    if not want_J and not want_K:
        return None, None, None, None

    import cupy as cp  # noqa: PLC0415

    _ext = cueri_kernels.require_ext()
    from asuka.cueri.eri_dispatch import KernelBatch, run_kernel_batch_spd  # noqa: PLC0415
    from asuka.cueri.tasks import TaskList  # noqa: PLC0415

    nao = ctx.nao
    base_threads = int(ctx.threads)
    eval_threads = int(getattr(ctx, "eval_threads", base_threads))
    contract_threads = int(getattr(ctx, "contract_threads", base_threads))

    Da_gpu = cp.asarray(Da, dtype=cp.float64)
    if Da_gpu.ndim != 2 or Da_gpu.shape != (nao, nao):
        raise ValueError(f"Da must be ({nao}, {nao}), got {tuple(Da_gpu.shape)}")
    Db_gpu = cp.asarray(Db, dtype=cp.float64)
    if Db_gpu.ndim != 2 or Db_gpu.shape != (nao, nao):
        raise ValueError(f"Db must be ({nao}, {nao}), got {tuple(Db_gpu.shape)}")

    Da_flat = cp.ascontiguousarray(Da_gpu.ravel())
    Db_flat = cp.ascontiguousarray(Db_gpu.ravel())

    nao2 = nao * nao
    Ja_flat = cp.zeros((nao2,), dtype=cp.float64) if want_J else None
    Ka_flat = cp.zeros((nao2,), dtype=cp.float64) if want_K else None
    Jb_flat = cp.zeros((nao2,), dtype=cp.float64) if want_J else None
    Kb_flat = cp.zeros((nao2,), dtype=cp.float64) if want_K else None

    compute_stream = cp.cuda.get_current_stream()
    stream_ptr = int(compute_stream.ptr)
    ws = _get_direct_jk_workspace(cp, ctx)
    n_kernel_calls = 0
    t0 = time.perf_counter()
    hot_s1_env = str(os.environ.get("ASUKA_DENSE_CONTRACT_HOT_S1", "") or "").strip().lower()
    hot_s1_contract_enabled = hot_s1_env not in {"0", "false", "off", "no"}
    eri_block_pref_env = str(os.environ.get("ASUKA_DIRECT_JK_ERI_BLOCK_MODE", "") or "").strip().lower()
    if eri_block_pref_env:
        eri_block_pref = eri_block_pref_env in {"1", "true", "on", "yes"}
    else:
        eri_block_pref = False

    if stats is not None:
        stats["direct_jk_ntasks"] = int(ctx.ntasks)
        stats.setdefault("resident_ntasks", 0)
        stats.setdefault("streamed_ntasks", 0)
        stats.setdefault("bytes_uploaded", 0)
        stats.setdefault("n_eval_calls", 0)
        stats.setdefault("n_contract_calls", 0)
        stats.setdefault("n_fused_calls", 0)
        stats.setdefault("classes", {})

    buf_slab: list[int | None] = [None, None]
    cur_buf = 0

    def _find_next_cpu(i: int) -> int | None:
        for j in range(int(i) + 1, int(len(ctx.slabs))):
            if not bool(ctx.slabs[j].gpu_resident):
                return int(j)
        return None

    def _am_label(l: int) -> str:
        am = "spdfghijklm"
        li = int(l)
        if 0 <= li < len(am):
            return am[li]
        return f"l{li}"

    def _class_label(cid: int) -> str:
        x = int(cid) & 0xFFFFFFFF
        la = x & 0xFF
        lb = (x >> 8) & 0xFF
        lc = (x >> 16) & 0xFF
        ld = (x >> 24) & 0xFF
        return f"{_am_label(la)}{_am_label(lb)}{_am_label(lc)}{_am_label(ld)}"

    def _enqueue_upload(slab_idx: int, buf_idx: int) -> None:
        assert ws is not None
        # Ensure we don't overwrite a buffer that compute is still reading.
        ws.copy_stream.wait_event(ws.compute_done[int(buf_idx)])

        slab_u = ctx.slabs[int(slab_idx)]
        nt = int(slab_u.ntasks)
        if nt <= 0:
            buf_slab[int(buf_idx)] = int(slab_idx)
            ws.upload_done[int(buf_idx)].record(ws.copy_stream)
            return
        if nt > int(ws.max_ntasks):
            raise RuntimeError(
                f"DirectJKWorkspace upload buffer too small: need ntasks={nt}, have {int(ws.max_ntasks)}"
            )
        ab_host = np.asarray(slab_u.ab_sorted, dtype=np.int32, order="C")
        cd_host = np.asarray(slab_u.cd_sorted, dtype=np.int32, order="C")
        dst_ab = ws.upload_ab[int(buf_idx)]
        dst_cd = ws.upload_cd[int(buf_idx)]
        stream_h2d = int(ws.copy_stream.ptr)
        cp.cuda.runtime.memcpyAsync(
            int(dst_ab.data.ptr),
            int(ab_host.ctypes.data),
            int(nt * 4),
            cp.cuda.runtime.memcpyHostToDevice,
            stream_h2d,
        )
        cp.cuda.runtime.memcpyAsync(
            int(dst_cd.data.ptr),
            int(cd_host.ctypes.data),
            int(nt * 4),
            cp.cuda.runtime.memcpyHostToDevice,
            stream_h2d,
        )
        ws.upload_done[int(buf_idx)].record(ws.copy_stream)
        buf_slab[int(buf_idx)] = int(slab_idx)
        if stats is not None:
            stats["bytes_uploaded"] = int(stats.get("bytes_uploaded", 0)) + int(nt * 8)

    for slab_i, slab in enumerate(ctx.slabs):
        slab_i = int(slab_i)

        # Prefetch the next CPU slab during GPU-resident slab compute when possible.
        if ws is not None and bool(slab.gpu_resident):
            nxt = _find_next_cpu(slab_i)
            if nxt is not None and nxt not in buf_slab:
                free_buf = 0 if buf_slab[0] is None else (1 if buf_slab[1] is None else None)
                if free_buf is not None:
                    _enqueue_upload(int(nxt), int(free_buf))

        used_buf: int | None = None
        if slab.gpu_resident:
            ab_dev = slab.ab_sorted
            cd_dev = slab.cd_sorted
            if stats is not None:
                stats["resident_ntasks"] = int(stats.get("resident_ntasks", 0)) + int(slab.ntasks)
        else:
            if stats is not None:
                stats["streamed_ntasks"] = int(stats.get("streamed_ntasks", 0)) + int(slab.ntasks)
            if ws is None:
                # Fallback: synchronous upload.
                ab_dev = cp.ascontiguousarray(cp.asarray(slab.ab_sorted, dtype=cp.int32))
                cd_dev = cp.ascontiguousarray(cp.asarray(slab.cd_sorted, dtype=cp.int32))
                if stats is not None:
                    stats["bytes_uploaded"] = int(stats.get("bytes_uploaded", 0)) + int(slab.ntasks * 8)
            else:
                # Ensure current slab is uploaded into one of the two buffers.
                if slab_i in buf_slab:
                    cur_buf = int(buf_slab.index(slab_i))
                else:
                    if buf_slab[int(cur_buf)] is not None:
                        cur_buf = 1 - int(cur_buf)
                    _enqueue_upload(int(slab_i), int(cur_buf))

                # Prefetch the next CPU slab into the other buffer (copy stream).
                nxt = _find_next_cpu(slab_i)
                other = 1 - int(cur_buf)
                if nxt is not None and nxt not in buf_slab and buf_slab[int(other)] is None:
                    _enqueue_upload(int(nxt), int(other))

                # Wait for H2D completion on the compute stream.
                compute_stream.wait_event(ws.upload_done[int(cur_buf)])
                nt = int(slab.ntasks)
                ab_dev = ws.upload_ab[int(cur_buf)][:nt]
                cd_dev = ws.upload_cd[int(cur_buf)][:nt]
                used_buf = int(cur_buf)

        slab_plan = ctx.plans[slab_i]
        for gp in slab_plan:
            orig_cid = int(gp.orig_cid)
            j0, j1 = int(gp.j0), int(gp.j1)
            if j1 <= j0:
                continue

            kernel_cid = int(gp.kernel_cid)
            transpose = bool(gp.transpose)
            nA = int(gp.nA)
            nB = int(gp.nB)
            nC = int(gp.nC)
            nD = int(gp.nD)
            chunk_ntasks = int(gp.chunk_ntasks)
            use_warp_mode = bool(gp.use_warp_eri)
            use_warp_contract = bool(gp.use_warp_contract)

            kernel_spAB_full = cd_dev[j0:j1] if transpose else ab_dev[j0:j1]
            kernel_spCD_full = ab_dev[j0:j1] if transpose else cd_dev[j0:j1]

            class_ntasks = j1 - j0
            if stats is not None:
                cls = _class_label(orig_cid)
                row = stats.setdefault("classes", {}).setdefault(cls, {"ntasks": 0, "chunks": 0, "path": ""})
                row["ntasks"] = int(row.get("ntasks", 0)) + int(class_ntasks)
                row["chunks"] = int(row.get("chunks", 0)) + int((class_ntasks + chunk_ntasks - 1) // chunk_ntasks)
                hot_s1 = bool(
                    (int(nB) == 1 and int(nD) == 1)
                    and (int(nA) in (1, 3, 6))
                    and (int(nC) in (1, 3, 6))
                )
                path = "staged_hot_s1_multi2" if (hot_s1 and hot_s1_contract_enabled) else "staged"
                prev = str(row.get("path", "") or "")
                row["path"] = path if (not prev or prev == path) else "mixed"

            for c0 in range(0, class_ntasks, chunk_ntasks):
                c1 = min(class_ntasks, c0 + chunk_ntasks)
                if c1 <= c0:
                    continue

                sub_batch = KernelBatch(
                    task_idx=np.empty(0, dtype=np.int32),  # unused by run_kernel_batch_spd
                    kernel_tasks=TaskList(
                        task_spAB=kernel_spAB_full[c0:c1],
                        task_spCD=kernel_spCD_full[c0:c1],
                    ),
                    kernel_class_id=np.int32(kernel_cid),
                    transpose=transpose,
                )

                tiles = run_kernel_batch_spd(
                    sub_batch,
                    dbasis=ctx.dbasis,
                    dsp=ctx.dsp,
                    pt=ctx.pair_tables,
                    stream=None,
                    threads=eval_threads,
                    mode="warp" if use_warp_mode else ("block" if eri_block_pref else "auto"),
                    profile=profile,
                    skip_transpose=True,
                )
                n_kernel_calls += 1
                if stats is not None:
                    stats["n_eval_calls"] = int(stats.get("n_eval_calls", 0)) + 1

                if profile is not None:
                    _tc0 = cp.cuda.Event()
                    _tc1 = cp.cuda.Event()
                    _tc0.record()

                contract_fn = (
                    _ext.contract_jk_tiles_ordered_warp_multi2_inplace_device
                    if use_warp_contract
                    else _ext.contract_jk_tiles_ordered_multi2_inplace_device
                )

                if transpose:
                    contract_fn(
                        cd_dev[j0 + c0: j0 + c1],
                        ab_dev[j0 + c0: j0 + c1],
                        ctx.sp_A_dev,
                        ctx.sp_B_dev,
                        ctx.shell_ao_start_dev,
                        int(nao),
                        int(nC),
                        int(nD),
                        int(nA),
                        int(nB),
                        tiles.ravel(),
                        Da_flat,
                        Db_flat,
                        Ja_flat,
                        Ka_flat,
                        Jb_flat,
                        Kb_flat,
                        int(contract_threads),
                        int(stream_ptr),
                        False,
                    )
                else:
                    contract_fn(
                        ab_dev[j0 + c0: j0 + c1],
                        cd_dev[j0 + c0: j0 + c1],
                        ctx.sp_A_dev,
                        ctx.sp_B_dev,
                        ctx.shell_ao_start_dev,
                        int(nao),
                        int(nA),
                        int(nB),
                        int(nC),
                        int(nD),
                        tiles.ravel(),
                        Da_flat,
                        Db_flat,
                        Ja_flat,
                        Ka_flat,
                        Jb_flat,
                        Kb_flat,
                        int(contract_threads),
                        int(stream_ptr),
                        False,
                    )
                if stats is not None:
                    stats["n_contract_calls"] = int(stats.get("n_contract_calls", 0)) + 1

                if profile is not None:
                    _tc1.record()
                    _tc1.synchronize()
                    profile["contract_ms"] = float(profile.get("contract_ms", 0.0)) + float(
                        cp.cuda.get_elapsed_time(_tc0, _tc1)
                    )

        if used_buf is not None and ws is not None:
            # Prevent buffer reuse until all compute that reads it is complete.
            ws.compute_done[int(used_buf)].record(compute_stream)
            if slab_i in buf_slab:
                bi = int(buf_slab.index(slab_i))
                buf_slab[bi] = None

    Ja = Jb = Ka = Kb = None
    if Ja_flat is not None:
        Ja = 0.5 * (Ja_flat.reshape((nao, nao)) + Ja_flat.reshape((nao, nao)).T)
    if Ka_flat is not None:
        Ka = 0.5 * (Ka_flat.reshape((nao, nao)) + Ka_flat.reshape((nao, nao)).T)
    if Jb_flat is not None:
        Jb = 0.5 * (Jb_flat.reshape((nao, nao)) + Jb_flat.reshape((nao, nao)).T)
    if Kb_flat is not None:
        Kb = 0.5 * (Kb_flat.reshape((nao, nao)) + Kb_flat.reshape((nao, nao)).T)

    if profile is not None:
        profile["direct_jk_t_s"] = float(time.perf_counter() - t0)
        profile["direct_jk_kernel_calls"] = int(n_kernel_calls)
        profile["direct_jk_ntasks"] = int(ctx.ntasks)

    return Ja, Ka, Jb, Kb


__all__ = [
    "DirectJKContext",
    "DirectJKWorkspace",
    "release_direct_jk_workspace_cache",
    "direct_JK",
    "direct_JK_multi",
    "direct_fock_rhf",
    "make_direct_jk_context",
]
