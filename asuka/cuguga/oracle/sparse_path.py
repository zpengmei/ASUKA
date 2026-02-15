from __future__ import annotations

from collections import defaultdict

import numpy as np

from asuka.integrals.df_integrals import DFMOIntegrals
from asuka.cuguga.drt import DRT
from asuka.cuguga.epq.action import epq_contribs_one_keys, path_nodes
from asuka.cuguga.oracle import _STEP_TO_OCC, _restore_eri_4d
from asuka.cuguga.screening import RowScreening

PathKey = bytes


def _as_f64_square(a: np.ndarray, n: int, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.shape != (int(n), int(n)):
        raise ValueError(f"{name} has wrong shape: {arr.shape} (expected {(int(n), int(n))})")
    return arr


def _sorted_keys(acc: dict[PathKey, float], *, ket_key: PathKey) -> list[PathKey]:
    if ket_key not in acc:
        return sorted(acc.keys())
    others = sorted(k for k in acc.keys() if k != ket_key)
    return [ket_key] + others


def _coalesce_rs(rs_ids: np.ndarray, rs_coeff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if rs_ids.size <= 1:
        return rs_ids, rs_coeff
    order = np.argsort(rs_ids, kind="mergesort")
    rs_ids = rs_ids[order]
    rs_coeff = rs_coeff[order]
    change = np.nonzero(rs_ids[1:] != rs_ids[:-1])[0] + 1
    if change.size == 0:
        return rs_ids[:1], np.asarray([float(rs_coeff.sum())], dtype=np.float64)
    starts = np.concatenate(([0], change)).astype(np.int32, copy=False)
    return rs_ids[starts], np.add.reduceat(rs_coeff, starts)


def connected_row_sparse_by_path(
    drt: DRT,
    h1e: np.ndarray,
    eri,
    ket_steps: np.ndarray,
    *,
    max_out: int = 200_000,
    screening: RowScreening | None = None,
) -> tuple[list[PathKey], np.ndarray]:
    """Dense-ERI sparse row oracle returning neighbors as path keys (bytes).
    """

    max_out = int(max_out)
    if max_out < 1:
        raise ValueError("max_out must be >= 1")
    screen = RowScreening() if screening is None else screening

    norb = int(drt.norb)
    h1e = _as_f64_square(h1e, norb, "h1e")
    eri4 = _restore_eri_4d(eri, norb).astype(np.float64, copy=False)

    steps_j = np.asarray(ket_steps, dtype=np.int8).ravel()
    if int(steps_j.size) != norb:
        raise ValueError("ket_steps have wrong length for this DRT")
    nodes_j = path_nodes(drt, steps_j)
    ket_key = steps_j.tobytes()
    occ_j = _STEP_TO_OCC[steps_j].astype(np.int8, copy=False)

    h_eff = h1e - 0.5 * np.einsum("pqqs->ps", eri4, optimize=True)
    h_eff = h_eff + 0.5 * np.einsum("pqrr,r->pq", eri4, occ_j, optimize=True)

    acc: dict[PathKey, float] = defaultdict(float)
    acc[ket_key] += float(np.dot(np.diag(h_eff), occ_j.astype(np.float64)))

    src1 = np.nonzero(occ_j > 0)[0].tolist()
    dst1 = np.nonzero(occ_j < 2)[0].tolist()
    for q in src1:
        for p in dst1:
            if p == q:
                continue
            hpq = float(h_eff[int(p), int(q)])
            if hpq == 0.0 or abs(hpq) <= float(screen.thresh_h1e):
                continue
            i_keys, coeff = epq_contribs_one_keys(drt, int(p), int(q), steps=steps_j, nodes=nodes_j)
            for key, cc in zip(i_keys, coeff):
                val = hpq * float(cc)
                if abs(val) <= float(screen.thresh_contrib):
                    continue
                acc[key] += val

    # Two-body via intermediate states k = E_rs |j>.
    nops = norb * norb
    eri_mat = eri4.reshape(nops, nops)

    by_k: dict[PathKey, list[tuple[int, float]]] = {}
    for s in src1:
        for r in dst1:
            if r == s:
                continue
            k_keys, c_rs = epq_contribs_one_keys(drt, int(r), int(s), steps=steps_j, nodes=nodes_j)
            if not k_keys:
                continue
            rs_id = int(r) * norb + int(s)
            for kk, cc in zip(k_keys, c_rs):
                by_k.setdefault(kk, []).append((rs_id, float(cc)))

    for k_key, rs_terms in by_k.items():
        rs_ids = np.asarray([t[0] for t in rs_terms], dtype=np.int32)
        rs_coeff = np.asarray([t[1] for t in rs_terms], dtype=np.float64)
        rs_ids, rs_coeff = _coalesce_rs(rs_ids, rs_coeff)
        if float(screen.thresh_rs_coeff) > 0.0:
            keep = np.abs(rs_coeff) > float(screen.thresh_rs_coeff)
            rs_ids = rs_ids[keep]
            rs_coeff = rs_coeff[keep]
            if rs_ids.size == 0:
                continue

        g_flat = 0.5 * (eri_mat[:, rs_ids] @ rs_coeff)
        g = g_flat.reshape(norb, norb)

        steps_k = np.frombuffer(k_key, dtype=np.int8)
        nodes_k = path_nodes(drt, steps_k)
        occ_k = _STEP_TO_OCC[steps_k].astype(np.int8, copy=False)

        diag_g = np.diag(g)
        if float(screen.thresh_gpq) > 0.0:
            mask = np.abs(diag_g) > float(screen.thresh_gpq)
            diag_contrib = float(np.dot(diag_g[mask], occ_k[mask].astype(np.float64)))
        else:
            diag_contrib = float(np.dot(diag_g, occ_k.astype(np.float64)))
        if abs(diag_contrib) > float(screen.thresh_contrib):
            acc[k_key] += diag_contrib

        src_k = np.nonzero(occ_k > 0)[0].tolist()
        dst_k = np.nonzero(occ_k < 2)[0].tolist()
        for q in src_k:
            for p in dst_k:
                if p == q:
                    continue
                gpq = float(g[int(p), int(q)])
                if gpq == 0.0 or abs(gpq) <= float(screen.thresh_gpq):
                    continue
                i_keys, coeff = epq_contribs_one_keys(drt, int(p), int(q), steps=steps_k, nodes=nodes_k)
                for key, cc in zip(i_keys, coeff):
                    val = gpq * float(cc)
                    if abs(val) <= float(screen.thresh_contrib):
                        continue
                    acc[key] += val

    keys = _sorted_keys(acc, ket_key=ket_key)
    hij = np.asarray([float(acc[k]) for k in keys], dtype=np.float64)

    if len(keys) > max_out:
        raise ValueError(f"oracle produced {len(keys)} entries > max_out={max_out}")
    return keys, hij


def connected_row_sparse_df_by_path(
    drt: DRT,
    h1e: np.ndarray,
    df_eri: DFMOIntegrals,
    ket_steps: np.ndarray,
    *,
    max_out: int = 200_000,
    screening: RowScreening | None = None,
) -> tuple[list[PathKey], np.ndarray]:
    """DF-backed sparse row oracle returning neighbors as path keys (bytes)."""

    max_out = int(max_out)
    if max_out < 1:
        raise ValueError("max_out must be >= 1")
    screen = RowScreening() if screening is None else screening

    norb = int(drt.norb)
    if int(df_eri.norb) != norb:
        raise ValueError(f"df_eri.norb={int(df_eri.norb)} does not match drt.norb={norb}")
    h1e = _as_f64_square(h1e, norb, "h1e")

    steps_j = np.asarray(ket_steps, dtype=np.int8).ravel()
    if int(steps_j.size) != norb:
        raise ValueError("ket_steps have wrong length for this DRT")
    nodes_j = path_nodes(drt, steps_j)
    ket_key = steps_j.tobytes()
    occ_j = _STEP_TO_OCC[steps_j].astype(np.int8, copy=False)

    h_eff = h1e - 0.5 * np.asarray(df_eri.j_ps, dtype=np.float64)
    h_eff = h_eff + df_eri.rr_slice_h_eff(occ_j, half=0.5, eri_mat_max_bytes=int(screen.df_eri_mat_max_bytes))

    acc: dict[PathKey, float] = defaultdict(float)
    acc[ket_key] += float(np.dot(np.diag(h_eff), occ_j.astype(np.float64)))

    src1 = np.nonzero(occ_j > 0)[0].tolist()
    dst1 = np.nonzero(occ_j < 2)[0].tolist()
    for q in src1:
        for p in dst1:
            if p == q:
                continue
            hpq = float(h_eff[int(p), int(q)])
            if hpq == 0.0 or abs(hpq) <= float(screen.thresh_h1e):
                continue
            i_keys, coeff = epq_contribs_one_keys(drt, int(p), int(q), steps=steps_j, nodes=nodes_j)
            for key, cc in zip(i_keys, coeff):
                val = hpq * float(cc)
                if abs(val) <= float(screen.thresh_contrib):
                    continue
                acc[key] += val

    by_k: dict[PathKey, list[tuple[int, float]]] = {}
    for s in src1:
        for r in dst1:
            if r == s:
                continue
            k_keys, c_rs = epq_contribs_one_keys(drt, int(r), int(s), steps=steps_j, nodes=nodes_j)
            if not k_keys:
                continue
            rs_id = int(r) * norb + int(s)
            for kk, cc in zip(k_keys, c_rs):
                by_k.setdefault(kk, []).append((rs_id, float(cc)))

    for k_key, rs_terms in by_k.items():
        rs_ids = np.asarray([t[0] for t in rs_terms], dtype=np.int32)
        rs_coeff = np.asarray([t[1] for t in rs_terms], dtype=np.float64)
        rs_ids, rs_coeff = _coalesce_rs(rs_ids, rs_coeff)
        if float(screen.thresh_rs_coeff) > 0.0:
            keep = np.abs(rs_coeff) > float(screen.thresh_rs_coeff)
            rs_ids = rs_ids[keep]
            rs_coeff = rs_coeff[keep]
            if rs_ids.size == 0:
                continue
        if float(screen.thresh_rs_pairnorm) > 0.0:
            keep = np.abs(rs_coeff) * df_eri.pair_norm[rs_ids] > float(screen.thresh_rs_pairnorm)
            rs_ids = rs_ids[keep]
            rs_coeff = rs_coeff[keep]
            if rs_ids.size == 0:
                continue

        g_flat = df_eri.contract_cols(
            rs_ids, rs_coeff, half=0.5, eri_mat_max_bytes=int(screen.df_eri_mat_max_bytes)
        )
        g = g_flat.reshape(norb, norb)

        steps_k = np.frombuffer(k_key, dtype=np.int8)
        nodes_k = path_nodes(drt, steps_k)
        occ_k = _STEP_TO_OCC[steps_k].astype(np.int8, copy=False)

        diag_g = np.diag(g)
        if float(screen.thresh_gpq) > 0.0:
            mask = np.abs(diag_g) > float(screen.thresh_gpq)
            diag_contrib = float(np.dot(diag_g[mask], occ_k[mask].astype(np.float64)))
        else:
            diag_contrib = float(np.dot(diag_g, occ_k.astype(np.float64)))
        if abs(diag_contrib) > float(screen.thresh_contrib):
            acc[k_key] += diag_contrib

        src_k = np.nonzero(occ_k > 0)[0].tolist()
        dst_k = np.nonzero(occ_k < 2)[0].tolist()
        for q in src_k:
            for p in dst_k:
                if p == q:
                    continue
                gpq = float(g[int(p), int(q)])
                if gpq == 0.0 or abs(gpq) <= float(screen.thresh_gpq):
                    continue
                i_keys, coeff = epq_contribs_one_keys(drt, int(p), int(q), steps=steps_k, nodes=nodes_k)
                for key, cc in zip(i_keys, coeff):
                    val = gpq * float(cc)
                    if abs(val) <= float(screen.thresh_contrib):
                        continue
                    acc[key] += val

    keys = _sorted_keys(acc, ket_key=ket_key)
    hij = np.asarray([float(acc[k]) for k in keys], dtype=np.float64)

    if len(keys) > max_out:
        raise ValueError(f"oracle produced {len(keys)} entries > max_out={max_out}")
    return keys, hij
