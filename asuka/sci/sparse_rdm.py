"""Sparse RDM computation for selected CI wavefunctions.

Computes 1-RDM and 2-RDM using only the nsel selected CSFs,
avoiding the O(ncsf) operations that would OOM for large spaces.

Uses the Cython-accelerated E_pq evaluator for speed.
Cost: O(nsel * norb^2 * branching) for E_pq evaluations.
Memory: O(norb^2 * nsel) for the T matrix.
"""

from __future__ import annotations

import numpy as np

from asuka.cuguga.drt import DRT

_STEP_TO_OCC = np.array([0, 1, 1, 2], dtype=np.int32)


def _decode_selected_csfs(drt: DRT, sel_idx: np.ndarray):
    """Decode step/node arrays for selected CSF indices only."""
    sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
    nsel = int(sel_idx.size)
    norb = int(drt.norb)
    child = np.asarray(drt.child, dtype=np.int32, order="C")
    nwalks = np.asarray(drt.nwalks, dtype=np.int64)
    root = int(drt.root)
    steps = np.empty((nsel, norb), dtype=np.int8, order="C")
    nodes = np.empty((nsel, norb + 1), dtype=np.int32, order="C")
    for i in range(nsel):
        idx = int(sel_idx[i])
        node = root
        nodes[i, 0] = node
        for k in range(norb):
            remaining = idx
            step = -1
            for s in range(4):
                c = int(child[node, s])
                if c < 0:
                    continue
                w = int(nwalks[c])
                if remaining < w:
                    step = s
                    node = c
                    break
                remaining -= w
            steps[i, k] = step
            nodes[i, k + 1] = node
            idx = remaining
    return steps, nodes


def make_rdm12_selected(
    drt: DRT,
    sel_idx: np.ndarray,
    ci_sel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (dm1, dm2) from a sparse CI vector over selected CSFs.

    Parameters
    ----------
    drt : DRT
    sel_idx : (nsel,) int64 — global CSF indices of selected states.
    ci_sel : (nsel,) float64 — CI coefficients.

    Returns
    -------
    dm1 : (norb, norb) float64
    dm2 : (norb, norb, norb, norb) float64
    """
    # Try Cython accelerated path
    try:
        from asuka.cuguga.oracle._cache import _epq_contribs_one_cy as _epq_fn
    except ImportError:
        from asuka.cuguga.oracle._cache import (
            _e_pq_contribs_from_csf_index_arrays as _epq_fn,
        )

    sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
    ci_sel = np.asarray(ci_sel, dtype=np.float64).ravel()
    nsel = int(sel_idx.size)
    norb = int(drt.norb)
    nops = norb * norb

    # Decode steps/nodes for selected CSFs only (NOT all ncsf)
    sel_steps, sel_nodes = _decode_selected_csfs(drt, sel_idx)

    # Build fast lookup: global CSF index -> local index
    global_to_local = {}
    for i in range(nsel):
        global_to_local[int(sel_idx[i])] = i

    # Build T[pq, nsel] = E_pq|c> projected onto selected basis
    T = np.zeros((nops, nsel), dtype=np.float64)

    for p in range(norb):
        for q in range(norb):
            pq = p * norb + q
            if p == q:
                occ_p = _STEP_TO_OCC[sel_steps[:, p].astype(np.int32)]
                T[pq, :] = occ_p.astype(np.float64) * ci_sel
                continue

            t_row = T[pq]
            for j_local in range(nsel):
                cj = ci_sel[j_local]
                if cj == 0.0:
                    continue
                j_global = int(sel_idx[j_local])
                steps_j = sel_steps[j_local]
                nodes_j = sel_nodes[j_local]
                occ_q = int(_STEP_TO_OCC[int(steps_j[q])])
                occ_p = int(_STEP_TO_OCC[int(steps_j[p])])
                if occ_q <= 0 or occ_p >= 2:
                    continue

                bra_idx, coeff = _epq_fn(
                    drt, j_global, p, q, steps_j, nodes_j,
                )
                if bra_idx.size == 0:
                    continue
                for k in range(bra_idx.size):
                    bra_local = global_to_local.get(int(bra_idx[k]), -1)
                    if bra_local >= 0:
                        t_row[bra_local] += float(coeff[k]) * cj

    # dm1 and Gram via BLAS
    dm1 = (T @ ci_sel).reshape(norb, norb).T
    gram0 = T @ T.T
    swap = np.arange(nops, dtype=np.int32).reshape(norb, norb).T.ravel()
    gram = gram0[swap]
    dm2 = gram.reshape(norb, norb, norb, norb).copy()
    for p in range(norb):
        for q in range(norb):
            dm2[p, q, q, :] -= dm1[:, p]
    return dm1, dm2
