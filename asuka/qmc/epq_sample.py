from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.epq.action import epq_contribs_one, path_nodes

try:  # optional compiled fast path (sample without building idx/coeff arrays)
    from asuka._epq_cy import epq_sample_from_csf_index_cy as _epq_sample_from_csf_index_cy  # type: ignore
except Exception:  # pragma: no cover
    _epq_sample_from_csf_index_cy = None


def _step_to_occ(step: int) -> int:
    if step == 0:
        return 0
    if step == 3:
        return 2
    return 1


@dataclass(frozen=True)
class EpqSample:
    child: int
    coeff: float
    inv_p: float
    valid: bool


def sample_epq_from_arrays(idx: np.ndarray, coeff: np.ndarray, rng: np.random.Generator) -> EpqSample:
    """Sample one child from an enumerated `E_pq|j>` action (CPU helper).

    Sampling distribution: p(i) ∝ |c_i|.

    Returns an unbiased single-term estimator for the full action via:
        contribution = coeff * inv_p  (placed on the sampled `child` index).
    """

    idx_i32 = np.asarray(idx, dtype=np.int32).ravel()
    coeff_f64 = np.asarray(coeff, dtype=np.float64).ravel()
    if idx_i32.size != coeff_f64.size:
        raise ValueError("idx and coeff must have the same size")
    if idx_i32.size == 0:
        return EpqSample(child=0, coeff=0.0, inv_p=0.0, valid=False)

    w = np.abs(coeff_f64)
    tot = float(np.sum(w))
    if tot == 0.0:
        return EpqSample(child=0, coeff=0.0, inv_p=0.0, valid=False)

    # Weighted pick with p(i) ∝ |c_i|, implemented via a CDF lookup.
    # This avoids `Generator.choice(p=...)`, which is relatively high overhead in tight loops.
    np.cumsum(w, out=w)
    u = float(rng.random()) * tot
    k = int(np.searchsorted(w, u, side="right"))
    if k >= int(idx_i32.size):
        k = int(idx_i32.size) - 1
    ck = float(coeff_f64[k])
    wk = float(abs(ck))
    if wk == 0.0:
        return EpqSample(child=0, coeff=0.0, inv_p=0.0, valid=False)
    return EpqSample(child=int(idx_i32[k]), coeff=ck, inv_p=(tot / wk), valid=True)


def sample_epq_one(
    drt: DRT,
    csf_idx: int,
    p: int,
    q: int,
    rng: np.random.Generator,
    *,
    steps: np.ndarray | None = None,
    nodes: np.ndarray | None = None,
) -> EpqSample:
    """Sample one child for `E_pq|csf_idx>` (CPU reference implementation)."""

    csf_idx = int(csf_idx)
    p = int(p)
    q = int(q)

    if steps is None:
        steps = drt.index_to_path(csf_idx)
    steps = np.asarray(steps, dtype=np.int8).ravel()

    if p == q:
        occ = _step_to_occ(int(steps[p]))
        if occ == 0:
            return EpqSample(child=0, coeff=0.0, inv_p=0.0, valid=False)
        return EpqSample(child=csf_idx, coeff=float(occ), inv_p=1.0, valid=True)

    if nodes is None:
        nodes = path_nodes(drt, steps)

    if _epq_sample_from_csf_index_cy is not None:
        child, coeff, inv_p, valid = _epq_sample_from_csf_index_cy(
            drt,
            int(csf_idx),
            int(p),
            int(q),
            steps,
            nodes,
            float(rng.random()),
        )
        return EpqSample(child=int(child), coeff=float(coeff), inv_p=float(inv_p), valid=bool(valid))

    idx, coeff = epq_contribs_one(drt, csf_idx, p, q, steps=steps, nodes=nodes)
    return sample_epq_from_arrays(idx, coeff, rng)
