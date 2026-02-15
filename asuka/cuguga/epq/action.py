from __future__ import annotations

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.oracle import (
    _Q_L,
    _Q_R,
    _Q_oL,
    _Q_oR,
    _Q_uL,
    _Q_uR,
    _STEP_TO_OCC,
    _e_pq_contribs_from_csf_index_arrays,
    _segment_value_int,
)

_BYTE_STEP: tuple[bytes, ...] = (b"\x00", b"\x01", b"\x02", b"\x03")

try:  # optional compiled fast path
    from asuka._epq_cy import epq_contribs_from_csf_index_arrays_cy as _epq_contribs_one_cy
except Exception:  # pragma: no cover
    _epq_contribs_one_cy = None

try:  # optional compiled fast path (batched apply)
    from asuka._epq_cy import epq_apply_weighted_many_cy as _epq_apply_weighted_many_cy
except Exception:  # pragma: no cover
    _epq_apply_weighted_many_cy = None

try:  # optional compiled fast path (apply g[p,q] matrix directly)
    from asuka._epq_cy import epq_apply_g_cy as _epq_apply_g_cy
except Exception:  # pragma: no cover
    _epq_apply_g_cy = None


def path_nodes(drt: DRT, steps: np.ndarray) -> np.ndarray:
    """Return the sequence of DRT node indices visited by a CSF walk.

    Parameters
    ----------
    drt : DRT
        Distinct Row Table.
    steps : np.ndarray
        Step vector of shape ``(norb,)`` with values in ``{0, 1, 2, 3}``.

    Returns
    -------
    np.ndarray
        int32 array of shape ``(norb + 1,)`` where ``nodes[0]`` is the root
        and ``nodes[norb]`` is the leaf.
    """
    steps = np.asarray(steps, dtype=np.int8).ravel()
    if steps.size != int(drt.norb):
        raise ValueError("steps have wrong length for this DRT")
    node = int(drt.root)
    norb = int(drt.norb)
    nodes = np.empty(norb + 1, dtype=np.int32)
    nodes[0] = node
    for k in range(norb):
        node = int(drt.child[node, int(steps[k])])
        if node < 0:
            raise ValueError("invalid path for this DRT")
        nodes[k + 1] = node
    return nodes


def epq_contribs_one(
    drt: DRT,
    csf_idx: int,
    p: int,
    q: int,
    *,
    steps: np.ndarray | None = None,
    nodes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the sparse action of E_pq on a single CSF.

    Thin wrapper around the segment-walk DFS in
    ``oracle._e_pq_contribs_from_csf_index_arrays``. Unlike the cached oracle
    path, this does **not** build a global CSR matrix over the full CSF basis.

    Parameters
    ----------
    drt : DRT
        Distinct Row Table.
    csf_idx : int
        Global CSF index of the bra/source CSF.
    p, q : int
        Orbital indices of the E_pq generator (``p != q``).
    steps : np.ndarray or None, optional
        Pre-computed step vector. Derived from *csf_idx* if *None*.
    nodes : np.ndarray or None, optional
        Pre-computed node sequence. Derived from *steps* if *None*.

    Returns
    -------
    idx : np.ndarray
        int32 array of destination CSF indices.
    coeff : np.ndarray
        float64 array of coupling coefficients.
    """

    csf_idx = int(csf_idx)
    if steps is None:
        steps = drt.index_to_path(csf_idx)
    if nodes is None:
        nodes = path_nodes(drt, steps)

    if _epq_contribs_one_cy is not None:
        idx, coeff = _epq_contribs_one_cy(drt, csf_idx, int(p), int(q), steps, nodes)
    else:
        idx, coeff = _e_pq_contribs_from_csf_index_arrays(
            drt,
            csf_idx,
            int(p),
            int(q),
            steps=steps,
            nodes=nodes,
        )
    return idx, coeff


def epq_apply_weighted_many(
    drt: DRT,
    csf_idx: int,
    p_idx: np.ndarray,
    q_idx: np.ndarray,
    weights: np.ndarray,
    *,
    steps: np.ndarray,
    nodes: np.ndarray,
    thresh_contrib: float = 0.0,
    trusted: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Σ_t weights[t] * E_{p_idx[t],q_idx[t]} to |csf_idx>.

    Returns concatenated COO arrays (i_idx, val) where:
      val = weights[t] * <i|E_pq|csf_idx>

    Notes
    -----
    - If the Cython extension is available, this uses a batched kernel to avoid
      per-(p,q) Python overhead and tiny-array allocations.
    - The fallback path loops over :func:`epq_contribs_one`.
    """

    csf_idx = int(csf_idx)
    p_idx = np.asarray(p_idx, dtype=np.int32).ravel()
    q_idx = np.asarray(q_idx, dtype=np.int32).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if p_idx.size != q_idx.size or p_idx.size != weights.size:
        raise ValueError("p_idx, q_idx, and weights must have the same length")

    if p_idx.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    steps = np.asarray(steps, dtype=np.int8).ravel()
    nodes = np.asarray(nodes, dtype=np.int32).ravel()

    if _epq_apply_weighted_many_cy is not None:
        return _epq_apply_weighted_many_cy(
            drt,
            csf_idx,
            p_idx,
            q_idx,
            weights,
            steps,
            nodes,
            float(thresh_contrib),
            bool(trusted),
        )

    # Fallback: do the same work with repeated epq_contribs_one calls.
    out_i: list[np.ndarray] = []
    out_v: list[np.ndarray] = []
    thresh = float(thresh_contrib)
    for p, q, w in zip(p_idx.tolist(), q_idx.tolist(), weights.tolist()):
        if w == 0.0:
            continue
        i_idx, coeff = epq_contribs_one(drt, csf_idx, int(p), int(q), steps=steps, nodes=nodes)
        if i_idx.size == 0:
            continue
        val = float(w) * np.asarray(coeff, dtype=np.float64)
        if thresh > 0.0:
            keep = np.abs(val) > thresh
            if not np.any(keep):
                continue
            out_i.append(np.asarray(i_idx[keep], dtype=np.int32, order="C"))
            out_v.append(np.asarray(val[keep], dtype=np.float64, order="C"))
        else:
            out_i.append(np.asarray(i_idx, dtype=np.int32, order="C"))
            out_v.append(np.asarray(val, dtype=np.float64, order="C"))

    if not out_i:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)
    return np.concatenate(out_i), np.concatenate(out_v)


def epq_apply_g(
    drt: DRT,
    csf_idx: int,
    g_flat: np.ndarray,
    *,
    steps: np.ndarray,
    nodes: np.ndarray,
    thresh_gpq: float = 0.0,
    thresh_contrib: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply Σ_{p,q} g[p,q] * E_pq to |csf_idx>, including diagonal p==q.

    This is a convenience wrapper used by the sparse row oracle to avoid
    allocating and filtering (p_idx, q_idx, weights) arrays in Python for every
    intermediate state.

    Parameters
    ----------
    g_flat
        Flattened (norb*norb,) array representing g[p,q] in row-major order.

    Returns
    -------
    i_idx, val, n_pairs
        COO arrays plus the number of off-diagonal (p,q) pairs processed.
    """

    csf_idx = int(csf_idx)
    g_flat = np.asarray(g_flat, dtype=np.float64).ravel(order="C")
    steps = np.asarray(steps, dtype=np.int8).ravel()
    nodes = np.asarray(nodes, dtype=np.int32).ravel()

    if _epq_apply_g_cy is not None:
        i_idx, val, n_pairs = _epq_apply_g_cy(
            drt,
            csf_idx,
            g_flat,
            steps,
            nodes,
            float(thresh_gpq),
            float(thresh_contrib),
        )
        return i_idx, val, int(n_pairs)

    # Slow fallback: select off-diagonal pairs in Python and reuse epq_apply_weighted_many.
    norb = int(drt.norb)
    if g_flat.size != norb * norb:
        raise ValueError("g_flat has wrong length for this DRT")

    tg = float(thresh_gpq)
    tc = float(thresh_contrib)

    # Diagonal contribution: Σ_p g[p,p] * occ[p]
    diag = 0.0
    for p in range(norb):
        gpp = float(g_flat[p * norb + p])
        if gpp == 0.0:
            continue
        if tg > 0.0 and abs(gpp) <= tg:
            continue
        occ_p = int(_STEP_TO_OCC[int(steps[p])])
        if occ_p:
            diag += gpp * float(occ_p)

    p_list: list[int] = []
    q_list: list[int] = []
    w_list: list[float] = []
    for p in range(norb):
        occ_p = int(_STEP_TO_OCC[int(steps[p])])
        if occ_p >= 2:
            continue
        for q in range(norb):
            if q == p:
                continue
            occ_q = int(_STEP_TO_OCC[int(steps[q])])
            if occ_q <= 0:
                continue
            gpq = float(g_flat[p * norb + q])
            if gpq == 0.0:
                continue
            if tg > 0.0 and abs(gpq) <= tg:
                continue
            p_list.append(int(p))
            q_list.append(int(q))
            w_list.append(float(gpq))

    if not p_list:
        if diag == 0.0 or (tc > 0.0 and abs(diag) <= tc):
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64), 0
        return np.asarray([csf_idx], dtype=np.int32), np.asarray([diag], dtype=np.float64), 0

    p_idx = np.asarray(p_list, dtype=np.int32)
    q_idx = np.asarray(q_list, dtype=np.int32)
    weights = np.asarray(w_list, dtype=np.float64)
    i_idx, val = epq_apply_weighted_many(
        drt,
        csf_idx,
        p_idx,
        q_idx,
        weights,
        steps=steps,
        nodes=nodes,
        thresh_contrib=tc,
        trusted=True,
    )
    if diag != 0.0 and not (tc > 0.0 and abs(diag) <= tc):
        i_idx = np.concatenate((np.asarray([csf_idx], dtype=np.int32), i_idx))
        val = np.concatenate((np.asarray([diag], dtype=np.float64), val))
    return i_idx, val, int(p_idx.size)


def epq_contribs_one_keys(
    drt: DRT,
    p: int,
    q: int,
    *,
    steps: np.ndarray,
    nodes: np.ndarray | None = None,
) -> tuple[list[bytes], np.ndarray]:
    """Return (neighbor_keys, coeff) for E_pq acting on a single CSF path.

    This is the path-native analogue of :func:`epq_contribs_one`. It returns
    neighbors as `PathKey = steps.tobytes()` blobs (length = norb bytes) instead
    of global CSF indices.
    """

    p = int(p)
    q = int(q)
    norb = int(drt.norb)
    if not (0 <= p < norb and 0 <= q < norb):
        raise ValueError("orbital indices out of range")
    if p == q:
        return [], np.zeros(0, dtype=np.float64)

    steps = np.asarray(steps, dtype=np.int8).ravel()
    if steps.size != norb:
        raise ValueError("steps have wrong length for this DRT")
    if nodes is None:
        nodes = path_nodes(drt, steps)

    # E_pq moves one electron from q to p: require occ[q] > 0 and occ[p] < 2.
    occ_p = int(_STEP_TO_OCC[int(steps[p])])
    occ_q = int(_STEP_TO_OCC[int(steps[q])])
    if occ_q <= 0 or occ_p >= 2:
        return [], np.zeros(0, dtype=np.float64)

    if p < q:
        start, end = p, q
        q_start, q_mid, q_end = _Q_uR, _Q_R, _Q_oR
    else:
        start, end = q, p
        q_start, q_mid, q_end = _Q_uL, _Q_L, _Q_oL

    node_start = int(nodes[start])
    node_end_target = int(nodes[end + 1])

    # Reference b_k values for each step k in [start, end].
    b_ref = drt.node_twos[nodes[start + 1 : end + 2]].astype(np.int32, copy=False)
    d_ref = steps[start : end + 1].astype(np.int8, copy=False)

    steps_bytes = steps.tobytes()
    prefix = steps_bytes[:start]
    suffix = steps_bytes[end + 1 :]

    out_keys: list[bytes] = []
    out_coeff: list[float] = []

    stack: list[tuple[int, int, float, bytes]] = [(start, node_start, 1.0, b"")]
    while stack:
        k, node_k, w, seg_bytes = stack.pop()
        is_first = k == start
        is_last = k == end
        qk = q_start if is_first else (q_end if is_last else q_mid)

        d_k = int(d_ref[k - start])
        b_k = int(b_ref[k - start])

        for dprime in range(4):
            child = int(drt.child[node_k, dprime])
            if child < 0:
                continue
            bprime = int(drt.node_twos[child])
            db = b_k - bprime
            seg = _segment_value_int(qk, dprime, d_k, db, b_k)
            if seg == 0.0:
                continue
            w2 = w * seg

            if is_last:
                if child != node_end_target:
                    continue
                key = prefix + seg_bytes + _BYTE_STEP[dprime] + suffix
                out_keys.append(key)
                out_coeff.append(float(w2))
            else:
                stack.append((k + 1, child, w2, seg_bytes + _BYTE_STEP[dprime]))

    return out_keys, np.asarray(out_coeff, dtype=np.float64)
