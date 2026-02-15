"""Caching infrastructure for E_pq generator actions and occupancy tables.

This module manages the precomputation and caching of:

- **Occupancy tables**: Per-CSF orbital occupancy (0/1/2) arrays.
- **Occupancy groups**: Compact occupancy-to-CSF-index mappings.
- **E_pq action caches**: CSR-format sparse matrices for generator actions.
- **Child prefix walks**: Cumulative walk counts used for fast CSF index
  recovery during the segment-walk DFS.

All caches use :class:`weakref.WeakKeyDictionary` keyed on :class:`DRT`
instances, so they are automatically garbage-collected when the DRT is no
longer referenced.
"""

from __future__ import annotations

import os
import weakref
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.oracle._segment import _segment_value_int

try:  # optional compiled fastpath for E_pq DFS expansion
    from asuka._epq_cy import (  # type: ignore[import-not-found]
        epq_contribs_from_csf_index_arrays_cy as _epq_contribs_one_cy,
    )
    try:
        from asuka._epq_cy import (  # type: ignore[import-not-found]
            epq_csc_for_pair_cy as _epq_csc_for_pair_cy,
        )
    except Exception:  # pragma: no cover
        _epq_csc_for_pair_cy = None
except Exception:  # pragma: no cover
    _epq_contribs_one_cy = None
    _epq_csc_for_pair_cy = None

# ---------------------------------------------------------------------------
# Step-to-occupancy mapping and global caches
# ---------------------------------------------------------------------------

_STEP_TO_OCC = np.asarray([0, 1, 1, 2], dtype=np.int8)  # E, U, L, D
_OCC_TABLE_CACHE: weakref.WeakKeyDictionary[DRT, np.ndarray] = weakref.WeakKeyDictionary()
_OCC_GROUPS_CACHE: weakref.WeakKeyDictionary[DRT, "OccGroups"] = weakref.WeakKeyDictionary()
_CHILD_PREFIX_WALKS_CACHE: weakref.WeakKeyDictionary[DRT, np.ndarray] = weakref.WeakKeyDictionary()


# ---------------------------------------------------------------------------
# CSR representation for E_pq actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CSR:
    """Compressed Sparse Row representation for a single E_pq operator."""

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray


@dataclass
class _EPQActionCache:
    """Cache holding precomputed steps/nodes tables and per-(p,q) CSR matrices.

    Attributes
    ----------
    steps : np.ndarray
        Shape ``(ncsf, norb)`` int8 step vectors for all CSFs.
    nodes : np.ndarray
        Shape ``(ncsf, norb+1)`` int32 node IDs along each CSF path.
    by_pair : dict[int, _CSR]
        Maps ``pair_id = p * norb + q`` to the CSR matrix for E_pq.
    """

    steps: np.ndarray
    nodes: np.ndarray
    by_pair: dict[int, _CSR]


_EPQ_ACTION_CACHE: weakref.WeakKeyDictionary[DRT, _EPQActionCache] = weakref.WeakKeyDictionary()


# ---------------------------------------------------------------------------
# Child prefix walks
# ---------------------------------------------------------------------------


def _child_prefix_walks(drt: DRT) -> np.ndarray:
    r"""Return cached prefix sums :math:`\sum_{prior < s} nwalks[child(node, prior)]`.

    Parameters
    ----------
    drt : DRT
        The Distinct Row Table.

    Returns
    -------
    np.ndarray
        Shape ``(nnodes, 5)`` int64 array where column ``s`` gives the
        cumulative walk count for steps prior to ``s`` at each node.
    """

    arr = _CHILD_PREFIX_WALKS_CACHE.get(drt)
    if arr is not None:
        return arr

    child = np.asarray(drt.child, dtype=np.int32)
    nwalks = np.asarray(drt.nwalks, dtype=np.int64)
    nnodes = int(child.shape[0])

    prefix = np.zeros((nnodes, 5), dtype=np.int64)
    for s in range(4):
        child_s = child[:, s]
        contrib = np.zeros(nnodes, dtype=np.int64)
        mask = child_s >= 0
        if np.any(mask):
            contrib[mask] = nwalks[child_s[mask]]
        prefix[:, s + 1] = prefix[:, s] + contrib

    prefix_c = np.asarray(prefix, order="C")
    _CHILD_PREFIX_WALKS_CACHE[drt] = prefix_c
    return prefix_c


# ---------------------------------------------------------------------------
# Occupancy tables and groups
# ---------------------------------------------------------------------------


def occ_table(drt: DRT) -> np.ndarray:
    """Return an ``(ncsf, norb)`` int8 occupancy table for the given DRT.

    Parameters
    ----------
    drt : DRT
        The Distinct Row Table.

    Returns
    -------
    np.ndarray
        Shape ``(ncsf, norb)`` array with values in {0, 1, 2} giving the
        orbital occupancy for each CSF.
    """

    cached = _OCC_TABLE_CACHE.get(drt)
    if cached is not None:
        return cached

    ncsf = int(drt.ncsf)
    occ = np.empty((ncsf, int(drt.norb)), dtype=np.int8)
    for i in range(ncsf):
        steps = drt.index_to_path(i)
        occ[i] = _STEP_TO_OCC[steps]
    _OCC_TABLE_CACHE[drt] = occ
    return occ


@dataclass(frozen=True)
class OccGroups:
    """Compact occupancy-to-CSF-index mapping.

    Groups CSF indices by their orbital occupation pattern for efficient
    lookup. Keys are ``bytes(occ_row)`` where ``occ_row`` is int8
    occupancy (0/1/2) of length ``norb``.

    Attributes
    ----------
    key_table : np.ndarray
        Sorted unique occupation keys (dtype void, ``norb`` bytes each).
    order : np.ndarray
        int32 CSF indices sorted by occupation key.
    offsets : np.ndarray
        int32 group boundaries into ``order`` (length ``nkeys + 1``).
    """

    key_table: np.ndarray  # dtype void (norb bytes), sorted
    order: np.ndarray  # int32, length ncsf, sorted by keys
    offsets: np.ndarray  # int32, length nkeys+1

    def __post_init__(self) -> None:
        if self.key_table.ndim != 1:
            raise ValueError("keys must be 1D")
        if self.order.ndim != 1:
            raise ValueError("order must be 1D")
        if self.offsets.ndim != 1:
            raise ValueError("offsets must be 1D")
        if self.offsets.size != self.key_table.size + 1:
            raise ValueError("offsets must have length len(keys)+1")
        if self.order.dtype != np.int32:
            raise ValueError("order must be int32")
        if self.offsets.dtype != np.int32:
            raise ValueError("offsets must be int32")

    @property
    def nbytes(self) -> int:
        """Total memory usage in bytes."""
        return int(self.key_table.nbytes) + int(self.order.nbytes) + int(self.offsets.nbytes)

    def _key_dtype(self) -> np.dtype:
        return np.dtype((np.void, int(self.key_table.dtype.itemsize)))

    def _find(self, key: bytes) -> int | None:
        if not isinstance(key, (bytes, bytearray, memoryview)):
            return None
        key = bytes(key)
        if len(key) != int(self.key_table.dtype.itemsize):
            return None
        dt = self._key_dtype()
        k = np.frombuffer(key, dtype=dt, count=1)[0]
        pos = int(np.searchsorted(self.key_table, k))
        if pos < int(self.key_table.size) and self.key_table[pos] == k:
            return pos
        return None

    def get(self, key: bytes, default=None):  # type: ignore[override]
        """Look up CSF indices for an occupation pattern.

        Parameters
        ----------
        key : bytes
            Occupation pattern as ``bytes(occ_row.tolist())``.
        default
            Value to return if the key is not found.

        Returns
        -------
        np.ndarray or default
            Slice of ``order`` covering all CSFs with this occupation,
            or ``default`` if not found.
        """
        pos = self._find(key)
        if pos is None:
            return default
        start = int(self.offsets[pos])
        end = int(self.offsets[pos + 1])
        return self.order[start:end]

    def __len__(self) -> int:
        return int(self.key_table.size)


def occ_groups(drt: DRT) -> OccGroups:
    """Return a compact occupancy-to-CSF-index mapping for the given DRT.

    Parameters
    ----------
    drt : DRT
        The Distinct Row Table.

    Returns
    -------
    OccGroups
        Mapping from occupation patterns to CSF indices.

    Notes
    -----
    This is an internal helper used by the "structure-only" row oracle.
    For large CSF spaces, a Python dict with one entry per unique occupation
    pattern is too slow/heavy. We therefore store a compact representation:

    - ``order``: indices sorted by occupation key
    - ``keys``: unique occupation keys (sorted)
    - ``offsets``: group boundaries into ``order``
    """

    cached = _OCC_GROUPS_CACHE.get(drt)
    if cached is not None:
        return cached

    occ = np.ascontiguousarray(occ_table(drt), dtype=np.int8)
    ncsf, norb = occ.shape

    # Represent each row as a fixed-width byte key (dtype void).
    key_dtype = np.dtype((np.void, int(norb)))
    keys_view = occ.view(key_dtype).reshape(ncsf)

    # Sort CSF indices by occupation key and find group boundaries.
    order = np.argsort(keys_view, kind="mergesort").astype(np.int32, copy=False)
    keys_sorted = keys_view[order]

    if ncsf:
        change = np.nonzero(keys_sorted[1:] != keys_sorted[:-1])[0] + 1
        offsets = np.concatenate(
            (
                np.asarray([0], dtype=np.int32),
                change.astype(np.int32, copy=False),
                np.asarray([ncsf], dtype=np.int32),
            )
        )
        uniq = keys_sorted[offsets[:-1]].copy()
    else:
        offsets = np.asarray([0], dtype=np.int32)
        uniq = keys_sorted.copy()

    out = OccGroups(key_table=uniq, order=order, offsets=offsets)
    _OCC_GROUPS_CACHE[drt] = out
    return out


# ---------------------------------------------------------------------------
# Path-node computation
# ---------------------------------------------------------------------------


def _path_nodes(drt: DRT, steps: np.ndarray) -> np.ndarray:
    """Compute the sequence of DRT node IDs visited by a step vector.

    Parameters
    ----------
    drt : DRT
        The Distinct Row Table.
    steps : np.ndarray
        Step vector of length ``norb`` with values in {0, 1, 2, 3}.

    Returns
    -------
    np.ndarray
        int32 array of length ``norb + 1`` giving node IDs from root to leaf.
    """
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


# ---------------------------------------------------------------------------
# E_pq contribs: segment-walk DFS
# ---------------------------------------------------------------------------


def _e_pq_contribs_from_csf_index(
    drt: DRT,
    csf_idx: int,
    p: int,
    q: int,
    *,
    steps: np.ndarray | None = None,
    nodes: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    """Return nonzero ``(i_idx, <i|E_pq|j>)`` for fixed CSF j and operator indices p, q.

    Parameters
    ----------
    drt : DRT
        The Distinct Row Table.
    csf_idx : int
        Index of the ket CSF ``|j>``.
    p, q : int
        Orbital indices for the generator E_pq.
    steps : np.ndarray, optional
        Precomputed step vector for CSF j.
    nodes : np.ndarray, optional
        Precomputed node sequence for CSF j.

    Returns
    -------
    list of (int, float)
        Pairs ``(i, coeff)`` where ``coeff = <i|E_pq|j>`` is nonzero.
    """

    idx, coeff = _e_pq_contribs_from_csf_index_arrays(
        drt,
        csf_idx,
        p,
        q,
        steps=steps,
        nodes=nodes,
    )
    return [(int(i), float(c)) for i, c in zip(idx.tolist(), coeff.tolist())]


def _e_pq_contribs_from_csf_index_arrays(
    drt: DRT,
    csf_idx: int,
    p: int,
    q: int,
    *,
    steps: np.ndarray | None = None,
    nodes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sparse ``(i_idx, <i|E_pq|j>)`` as arrays for fixed CSF j.

    This is the core segment-walk DFS algorithm. For each non-diagonal
    connection ``i != j``, it walks the DRT between orbitals ``min(p,q)``
    and ``max(p,q)``, accumulating the product of segment values along
    all valid alternative paths.

    Parameters
    ----------
    drt : DRT
        The Distinct Row Table.
    csf_idx : int
        Index of the ket CSF ``|j>``.
    p, q : int
        Orbital indices for the generator E_pq (must satisfy ``p != q``).
    steps : np.ndarray, optional
        Precomputed step vector for CSF j.
    nodes : np.ndarray, optional
        Precomputed node sequence for CSF j.

    Returns
    -------
    idx : np.ndarray
        int32 array of bra CSF indices with nonzero matrix elements.
    coeff : np.ndarray
        float64 array of corresponding ``<i|E_pq|j>`` values.
    """

    csf_idx = int(csf_idx)
    p = int(p)
    q = int(q)
    norb = int(drt.norb)
    if not (0 <= p < norb and 0 <= q < norb):
        raise ValueError("orbital indices out of range")
    if p == q:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    if steps is None:
        steps = drt.index_to_path(csf_idx).astype(np.int8, copy=False)
    if nodes is None:
        nodes = _path_nodes(drt, steps)

    # E_pq moves one electron from q to p: require occ[q] > 0 and occ[p] < 2.
    occ_p = int(_STEP_TO_OCC[int(steps[p])])
    occ_q = int(_STEP_TO_OCC[int(steps[q])])
    if occ_q <= 0 or occ_p >= 2:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    child = drt.child
    node_twos = drt.node_twos
    child_prefix = _child_prefix_walks(drt)

    if p < q:
        start, end = p, q
        q_start, q_mid, q_end = 1, 2, 3  # _Q_uR, _Q_R, _Q_oR
    else:
        start, end = q, p
        q_start, q_mid, q_end = 4, 5, 6  # _Q_uL, _Q_L, _Q_oL

    node_start = int(nodes[start])
    node_end_target = int(nodes[end + 1])

    # Reference b_k values for each step k in [start, end].
    b_ref = node_twos[nodes[start + 1 : end + 2]].astype(np.int32, copy=False)
    d_ref = steps[start : end + 1].astype(np.int8, copy=False)

    # Compute prefix indices for the reference path once, so we can recover each
    # neighbor CSF index without constructing a full `steps_p` and calling
    # `drt.path_to_index()` (which is O(norb) per neighbor).
    idx = 0
    prefix_offset = 0
    prefix_endplus1 = 0
    for kk in range(end + 1):
        if kk == start:
            prefix_offset = int(idx)
        node_kk = int(nodes[kk])
        step_kk = int(steps[kk])
        idx += int(child_prefix[node_kk, step_kk])
        if kk == end:
            prefix_endplus1 = int(idx)

    suffix_offset = int(csf_idx) - int(prefix_endplus1)

    idx_buf: list[int] = []
    coeff_buf: list[float] = []
    idx_append = idx_buf.append
    coeff_append = coeff_buf.append

    # Tuple-based stack is faster than SOA lists in CPython (fewer list ops).
    stack: list[tuple[int, int, float, int]] = [(start, node_start, 1.0, 0)]
    stack_pop = stack.pop
    stack_append = stack.append

    while stack:
        k, node_k, w, seg_idx = stack_pop()
        is_first = k == start
        is_last = k == end
        qk = q_start if is_first else (q_end if is_last else q_mid)

        d_k = int(d_ref[k - start])
        b_k = int(b_ref[k - start])
        k_next = k + 1

        for dprime in range(4):
            child_k = int(child[node_k, dprime])
            if child_k < 0:
                continue
            bprime = int(node_twos[child_k])
            db = b_k - bprime
            seg = _segment_value_int(qk, dprime, d_k, db, b_k)
            if seg == 0.0:
                continue
            w2 = w * seg

            seg_idx2 = seg_idx + int(child_prefix[node_k, dprime])

            if is_last:
                if child_k != node_end_target:
                    continue
                csf_i = prefix_offset + seg_idx2 + suffix_offset
                if w2 != 0.0 and csf_i != csf_idx:
                    idx_append(csf_i)
                    coeff_append(w2)
            else:
                stack_append((k_next, child_k, w2, seg_idx2))

    return np.asarray(idx_buf, dtype=np.int32), np.asarray(coeff_buf, dtype=np.float64)


# ---------------------------------------------------------------------------
# E_pq action cache management
# ---------------------------------------------------------------------------


def _get_epq_action_cache(drt: DRT) -> _EPQActionCache:
    """Get or build the E_pq action cache for a DRT.

    The cache stores step vectors and node sequences for all CSFs, enabling
    fast repeated E_pq evaluations without redundant ``index_to_path`` calls.

    Parameters
    ----------
    drt : DRT
        The Distinct Row Table.

    Returns
    -------
    _EPQActionCache
        Cache with ``steps``, ``nodes``, and ``by_pair`` CSR dict.
    """

    cached = _EPQ_ACTION_CACHE.get(drt)
    if cached is not None:
        return cached

    ncsf = int(drt.ncsf)
    norb = int(drt.norb)

    child = np.asarray(drt.child, dtype=np.int32, order="C")
    nwalks = np.asarray(drt.nwalks, dtype=np.int64)
    root = int(drt.root)
    leaf = int(drt.leaf)

    steps = np.empty((ncsf, norb), dtype=np.int8, order="C")
    nodes = np.empty((ncsf, norb + 1), dtype=np.int32, order="C")

    # Vectorized bulk index->path expansion (much faster than looping Python
    # `drt.index_to_path` over millions of CSFs).
    chunk_size = 200_000
    for start in range(0, ncsf, chunk_size):
        stop = min(ncsf, start + chunk_size)
        m = int(stop - start)
        idx = np.arange(start, stop, dtype=np.int64)
        node = np.full(m, root, dtype=np.int32)
        nodes[start:stop, 0] = node

        for k in range(norb):
            c0 = child[node, 0]
            c1 = child[node, 1]
            c2 = child[node, 2]
            c3 = child[node, 3]

            w0 = nwalks[c0]
            w1 = nwalks[c1]
            w2 = nwalks[c2]
            # Note: the DRT uses -1 for missing children; NumPy allows negative
            # indices, so explicitly zero their weights after the gather.
            m0 = c0 < 0
            if np.any(m0):
                w0[m0] = 0
            m1 = c1 < 0
            if np.any(m1):
                w1[m1] = 0
            m2 = c2 < 0
            if np.any(m2):
                w2[m2] = 0

            t0 = w0
            t1 = t0 + w1
            t2 = t1 + w2

            step = (idx >= t0).astype(np.int8) + (idx >= t1).astype(np.int8) + (idx >= t2).astype(np.int8)
            steps[start:stop, k] = step

            prefix = np.choose(step, [0, t0, t1, t2]).astype(np.int64, copy=False)
            idx = idx - prefix

            node = np.choose(step, [c0, c1, c2, c3]).astype(np.int32, copy=False)
            nodes[start:stop, k + 1] = node

        if not np.all(node == leaf):
            raise RuntimeError("index_to_path cache build did not terminate at target leaf")

    out = _EPQActionCache(steps=steps, nodes=nodes, by_pair={})
    _EPQ_ACTION_CACHE[drt] = out
    return out


def _csr_for_epq(cache: _EPQActionCache, drt: DRT, p: int, q: int) -> _CSR:
    """Get or build the CSR matrix for E_pq from the cache.

    Parameters
    ----------
    cache : _EPQActionCache
        The E_pq action cache for this DRT.
    drt : DRT
        The Distinct Row Table.
    p, q : int
        Orbital indices for the generator E_pq.

    Returns
    -------
    _CSR
        CSR matrix where row ``j`` stores the nonzero entries of ``<i|E_pq|j>``.
    """
    norb = int(drt.norb)
    pair_id = int(p) * norb + int(q)
    csr = cache.by_pair.get(pair_id)
    if csr is not None:
        return csr

    ncsf = int(drt.ncsf)

    if _epq_csc_for_pair_cy is not None:
        indptr, indices, data = _epq_csc_for_pair_cy(drt, int(p), int(q), cache.steps, cache.nodes)
        csr = _CSR(indptr=np.asarray(indptr, dtype=np.int32), indices=indices, data=data)
        cache.by_pair[pair_id] = csr
        return csr

    indptr = np.empty(ncsf + 1, dtype=np.int32)
    indptr[0] = 0

    # Avoid per-row `.tolist()` and Python list accumulation: grow NumPy buffers
    # dynamically and fill via slicing.
    nnz = 0
    cap = max(1024, ncsf * 2)
    indices = np.empty(cap, dtype=np.int32)
    data = np.empty(cap, dtype=np.float64)
    for csf_idx in range(ncsf):
        steps = cache.steps[csf_idx]
        nodes = cache.nodes[csf_idx]
        # Fast occupancy screening (avoid calling the DFS routine when the
        # operator is guaranteed to annihilate the state).
        occ_p = int(_STEP_TO_OCC[int(steps[p])])
        occ_q = int(_STEP_TO_OCC[int(steps[q])])
        if occ_q <= 0 or occ_p >= 2:
            indptr[csf_idx + 1] = int(nnz)
            continue
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
        k = int(idx.size)
        if k:
            need = nnz + k
            if need > cap:
                new_cap = max(need, cap * 2)
                new_indices = np.empty(new_cap, dtype=np.int32)
                new_data = np.empty(new_cap, dtype=np.float64)
                new_indices[:nnz] = indices[:nnz]
                new_data[:nnz] = data[:nnz]
                indices = new_indices
                data = new_data
                cap = int(new_cap)
            indices[nnz:need] = idx
            data[nnz:need] = coeff
            nnz = int(need)
        indptr[csf_idx + 1] = int(nnz)

    csr = _CSR(
        indptr=indptr,
        indices=np.asarray(indices[:nnz].copy(), dtype=np.int32),
        data=np.asarray(data[:nnz].copy(), dtype=np.float64),
    )
    cache.by_pair[pair_id] = csr
    return csr


def _epq_contribs_cached(
    cache: _EPQActionCache, drt: DRT, csf_idx: int, p: int, q: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return cached E_pq contribs for a single CSF from the CSR matrix.

    Parameters
    ----------
    cache : _EPQActionCache
        The E_pq action cache.
    drt : DRT
        The Distinct Row Table.
    csf_idx : int
        Index of the ket CSF ``|j>``.
    p, q : int
        Orbital indices.

    Returns
    -------
    idx : np.ndarray
        int32 bra CSF indices.
    coeff : np.ndarray
        float64 matrix element values.
    """
    if p == q:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)
    csr = _csr_for_epq(cache, drt, p, q)
    csf_idx = int(csf_idx)
    start = int(csr.indptr[csf_idx])
    end = int(csr.indptr[csf_idx + 1])
    return csr.indices[start:end], csr.data[start:end]


# ---------------------------------------------------------------------------
# Batch precomputation
# ---------------------------------------------------------------------------


def precompute_epq_actions(
    drt: DRT,
    *,
    nthreads: int = 1,
    pairs: list[tuple[int, int]] | None = None,
) -> None:
    """Precompute 1-body generator actions ``<i|E_pq|j>`` for this DRT.

    This trades one-time setup time and memory for fast repeated row
    generation in small-to-medium CSF spaces.

    Parameters
    ----------
    drt : DRT
        The Distinct Row Table.
    nthreads : int, optional
        If > 1 and the optional Cython fast path is available, build
        per-(p,q) CSC operators in parallel using a thread pool.
        Default is 1 (serial).
    pairs : list of (int, int), optional
        Orbital pairs ``(p, q)`` to build. If omitted, all off-diagonal
        pairs ``p != q`` are built.

    Notes
    -----
    The threaded build path has been observed to segfault intermittently
    (multiple concurrent calls into the Cython builder). It is opt-in
    via the ``CUGUGA_EPQ_PRECOMPUTE_PARALLEL`` environment variable.
    """

    cache = _get_epq_action_cache(drt)
    norb = int(drt.norb)
    nthreads = int(nthreads)
    if nthreads < 1:
        raise ValueError("nthreads must be >= 1")

    if pairs is None:
        want_pairs: list[tuple[int, int]] = [(p, q) for p in range(norb) for q in range(norb) if p != q]
    else:
        want_pairs = []
        seen: set[int] = set()
        for p, q in pairs:
            p = int(p)
            q = int(q)
            if p == q:
                continue
            if not (0 <= p < norb and 0 <= q < norb):
                raise ValueError("orbital indices out of range")
            pair_id = p * norb + q
            if pair_id in seen:
                continue
            seen.add(pair_id)
            want_pairs.append((p, q))

    # Fast parallel path only when the Cython builder exists. The pure-Python CSR
    # fallback does not benefit from threading (GIL-bound) and would add overhead.
    #
    # Note: The threaded build path has been observed to segfault intermittently
    # (multiple concurrent calls into the Cython builder). Keep it opt-in via an
    # env var until the underlying thread-safety issue is resolved.
    allow_parallel = os.environ.get("CUGUGA_EPQ_PRECOMPUTE_PARALLEL", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if nthreads > 1 and allow_parallel and _epq_csc_for_pair_cy is not None:
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        steps = cache.steps
        nodes = cache.nodes
        ncsf = int(drt.ncsf)
        if ncsf <= 0:
            return

        # Build missing pairs in parallel and commit to the shared cache dict
        # once complete (avoid concurrent dict mutation).
        todo: list[tuple[int, int, int]] = []
        for p, q in want_pairs:
            pair_id = int(p) * norb + int(q)
            if pair_id in cache.by_pair:
                continue
            todo.append((pair_id, int(p), int(q)))

        if not todo:
            return

        def _build_one(item: tuple[int, int, int]) -> tuple[int, _CSR]:
            pair_id, p, q = item
            indptr, indices, data = _epq_csc_for_pair_cy(drt, int(p), int(q), steps, nodes)
            csr = _CSR(indptr=np.asarray(indptr, dtype=np.int32), indices=indices, data=data)
            return int(pair_id), csr

        with ThreadPoolExecutor(max_workers=nthreads) as pool:
            for pair_id, csr in pool.map(_build_one, todo):
                cache.by_pair[int(pair_id)] = csr
        return

    if _epq_csc_for_pair_cy is not None:
        steps = cache.steps
        nodes = cache.nodes
        ncsf = int(drt.ncsf)
        if ncsf <= 0:
            return
        for p, q in want_pairs:
            pair_id = int(p) * norb + int(q)
            if pair_id in cache.by_pair:
                continue
            indptr, indices, data = _epq_csc_for_pair_cy(drt, int(p), int(q), steps, nodes)
            cache.by_pair[int(pair_id)] = _CSR(
                indptr=np.asarray(indptr, dtype=np.int32),
                indices=np.asarray(indices, dtype=np.int32),
                data=np.asarray(data, dtype=np.float64),
            )
        return

    for p, q in want_pairs:
        _csr_for_epq(cache, drt, int(p), int(q))


def epq_cache_nbytes(drt: DRT) -> int:
    """Return total bytes held by the E_pq action cache for this DRT (if built).

    Parameters
    ----------
    drt : DRT
        The Distinct Row Table.

    Returns
    -------
    int
        Total memory in bytes, or 0 if no cache exists.
    """

    cache = _EPQ_ACTION_CACHE.get(drt)
    if cache is None:
        return 0
    total = int(cache.steps.nbytes) + int(cache.nodes.nbytes)
    for csr in cache.by_pair.values():
        total += int(csr.indptr.nbytes) + int(csr.indices.nbytes) + int(csr.data.nbytes)
    return total
