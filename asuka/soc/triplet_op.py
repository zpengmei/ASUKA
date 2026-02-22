from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.oracle import _child_prefix_walks, _csr_for_epq, _get_epq_action_cache
from asuka.soc.triplet_factors import A_factor, Atilde_factor, B_factor, Btilde_factor, T_factor

_STEP_TO_OCC: Final[np.ndarray] = np.asarray([0, 1, 1, 2], dtype=np.int8)  # E, U, L, D


def _tri_ok_twos(tj1: int, tj2: int, tj3: int) -> bool:
    tj1 = int(tj1)
    tj2 = int(tj2)
    tj3 = int(tj3)
    if tj1 < 0 or tj2 < 0 or tj3 < 0:
        return False
    if (tj1 + tj2 + tj3) & 1:
        return False
    if tj3 < abs(tj1 - tj2) or tj3 > tj1 + tj2:
        return False
    return True


def rme_triplet_single_excitation(
    *,
    bra_steps: np.ndarray,
    ket_steps: np.ndarray,
    bra_twos_prefix: np.ndarray,
    ket_twos_prefix: np.ndarray,
    twos_bra_total: int,
    twos_ket_total: int,
    p: int,
    q: int,
) -> float:
    """Reduced <bra||T^(1)_{p q}||ket> for the triplet (rank-1) one-body operator.

    Implements the Lang et al. (JCTC 2025) factorized coupling-coefficient RMEs:
    - factor definitions: eqs. (63)–(67)
    - single-excitation RME assembly: eqs. (68)–(69)

    Phase-1 limitations
    -------------------
    - `p == q` (the special "annihilate and create in the same orbital" case, eq. 70)
      is not implemented yet (returns 0.0).
    """

    p = int(p)
    q = int(q)
    if p == q:
        return 0.0

    twos_bra_total = int(twos_bra_total)
    twos_ket_total = int(twos_ket_total)
    if not _tri_ok_twos(twos_bra_total, 2, twos_ket_total):
        return 0.0

    bra_steps = np.asarray(bra_steps, dtype=np.int8).ravel()
    ket_steps = np.asarray(ket_steps, dtype=np.int8).ravel()
    norb = int(ket_steps.size)
    if int(bra_steps.size) != norb:
        raise ValueError("bra_steps and ket_steps must have the same length")
    if p < 0 or q < 0 or p >= norb or q >= norb:
        raise IndexError("p/q out of range")

    bra_twos_prefix = np.asarray(bra_twos_prefix).ravel()
    ket_twos_prefix = np.asarray(ket_twos_prefix).ravel()
    if int(bra_twos_prefix.size) != norb + 1 or int(ket_twos_prefix.size) != norb + 1:
        raise ValueError("twos_prefix arrays must have length norb+1")

    occ_bra = _STEP_TO_OCC[bra_steps]
    occ_ket = _STEP_TO_OCC[ket_steps]
    diff_pos = np.nonzero(occ_bra != occ_ket)[0]
    if int(diff_pos.size) != 2:
        return 0.0

    # Reject double (or higher) occupancy changes at a single orbital.
    if int(np.max(np.abs(occ_bra[diff_pos] - occ_ket[diff_pos]))) != 1:
        return 0.0

    # Validate that the requested (p,q) matches the actual q->p excitation encoded by (bra,ket).
    if int(occ_bra[p]) != int(occ_ket[p]) + 1:
        return 0.0
    if int(occ_ket[q]) != int(occ_bra[q]) + 1:
        return 0.0
    if (int(diff_pos[0]) != min(p, q)) or (int(diff_pos[1]) != max(p, q)):
        # Required so the delta-prefix constraints align with the factorized expressions.
        return 0.0

    # Lang et al. eqs. (68)–(69) global prefactor: sqrt(3/2) / sqrt(2*S' + 1).
    coeff = math.sqrt(1.5) / math.sqrt(float(twos_bra_total + 1))

    # ζ_pq sign (Lang et al. Table 2, "complete sign", nonoverlapping cases).
    lo = min(p, q)
    hi = max(p, q)
    n_between = int(np.sum(occ_ket[lo + 1 : hi]))
    if p < q:
        # p < q case: add DOMO flags for created-orbital in bra and annihilated-orbital in ket.
        n_between += int(occ_bra[p] == 2) + int(occ_ket[q] == 2)
    coeff *= -1.0 if (n_between & 1) else 1.0

    # Kronecker deltas on prefix spins before the first orbital difference.
    # Paper uses orbital levels t=1..lo-1; in 0-based orbital indices this is prefix[1:lo].
    if lo > 0 and not np.array_equal(ket_twos_prefix[1:lo], bra_twos_prefix[1:lo]):
        return 0.0

    if p < q:
        # Eq. (68): p<q -> B_p, T(1/2) between, Ã_q(k=1), then T(k=1) after.
        p_occ_case = "SOMO" if int(occ_bra[p]) == 1 else "DOMO" if int(occ_bra[p]) == 2 else None
        if p_occ_case is None:
            return 0.0
        coeff *= float(
            B_factor(
                int(ket_twos_prefix[p]),
                int(ket_twos_prefix[p + 1]),
                int(bra_twos_prefix[p]),
                int(bra_twos_prefix[p + 1]),
                occ_case=p_occ_case,
            )
        )
        if coeff == 0.0:
            return 0.0

        for t in range(p + 1, q):
            if int(occ_ket[t]) != 1:
                continue
            coeff *= float(
                T_factor(
                    int(ket_twos_prefix[t]),
                    int(ket_twos_prefix[t + 1]),
                    int(bra_twos_prefix[t]),
                    int(bra_twos_prefix[t + 1]),
                    1,  # k=1/2
                )
            )
            if coeff == 0.0:
                return 0.0

        q_occ_case = "SOMO" if int(occ_ket[q]) == 1 else "DOMO" if int(occ_ket[q]) == 2 else None
        if q_occ_case is None:
            return 0.0
        coeff *= float(
            Atilde_factor(
                int(ket_twos_prefix[q]),
                int(ket_twos_prefix[q + 1]),
                int(bra_twos_prefix[q]),
                int(bra_twos_prefix[q + 1]),
                twos_opline=2,  # k=1
                occ_case=q_occ_case,
            )
        )
        if coeff == 0.0:
            return 0.0

        for t in range(q + 1, norb):
            if int(occ_ket[t]) != 1:
                continue
            coeff *= float(
                T_factor(
                    int(ket_twos_prefix[t]),
                    int(ket_twos_prefix[t + 1]),
                    int(bra_twos_prefix[t]),
                    int(bra_twos_prefix[t + 1]),
                    2,  # k=1
                )
            )
            if coeff == 0.0:
                return 0.0

        return float(coeff)

    # Eq. (69): q<p -> A_q, T(1/2) between, B̃_p(k=1), then T(k=1) after.
    q_occ_case = "SOMO" if int(occ_ket[q]) == 1 else "DOMO" if int(occ_ket[q]) == 2 else None
    if q_occ_case is None:
        return 0.0
    coeff *= float(
        A_factor(
            int(ket_twos_prefix[q]),
            int(ket_twos_prefix[q + 1]),
            int(bra_twos_prefix[q]),
            int(bra_twos_prefix[q + 1]),
            occ_case=q_occ_case,
        )
    )
    if coeff == 0.0:
        return 0.0

    for t in range(q + 1, p):
        if int(occ_ket[t]) != 1:
            continue
        coeff *= float(
            T_factor(
                int(ket_twos_prefix[t]),
                int(ket_twos_prefix[t + 1]),
                int(bra_twos_prefix[t]),
                int(bra_twos_prefix[t + 1]),
                1,  # k=1/2
            )
        )
        if coeff == 0.0:
            return 0.0

    p_occ_case = "SOMO" if int(occ_bra[p]) == 1 else "DOMO" if int(occ_bra[p]) == 2 else None
    if p_occ_case is None:
        return 0.0
    coeff *= float(
        Btilde_factor(
            int(ket_twos_prefix[p]),
            int(ket_twos_prefix[p + 1]),
            int(bra_twos_prefix[p]),
            int(bra_twos_prefix[p + 1]),
            twos_opline=2,  # k=1
            occ_case=p_occ_case,
        )
    )
    if coeff == 0.0:
        return 0.0

    for t in range(p + 1, norb):
        if int(occ_ket[t]) != 1:
            continue
        coeff *= float(
            T_factor(
                int(ket_twos_prefix[t]),
                int(ket_twos_prefix[t + 1]),
                int(bra_twos_prefix[t]),
                int(bra_twos_prefix[t + 1]),
                2,  # k=1
            )
        )
        if coeff == 0.0:
            return 0.0

    return float(coeff)


@dataclass
class TripletActionCache:
    steps: np.ndarray
    nodes: np.ndarray


def _get_triplet_action_cache(drt: DRT) -> TripletActionCache:
    # Reuse the E_pq cache (steps/nodes bulk expansion) for now.
    cache = _get_epq_action_cache(drt)
    return TripletActionCache(steps=cache.steps, nodes=cache.nodes)


def triplet_contribs_one(
    drt_bra: DRT,
    drt_ket: DRT,
    ket_idx: int,
    p: int,
    q: int,
    *,
    cache_bra: TripletActionCache | None = None,
    cache_ket: TripletActionCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (bra_indices, coeffs) for CSFs connected to ket_idx by T^(1)_{p q}.

    Same-DRT uses the cached scalar `E_pq` connectivity for indices.

    Cross-DRT uses a correctness-first DFS enumerator that enforces:
    - bra and ket steps match outside the interval `[min(p,q), max(p,q)]`
    - bra occupancy equals ket occupancy with one electron moved `q -> p`
    """

    p = int(p)
    q = int(q)
    if p == q:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    ket_idx = int(ket_idx)
    if ket_idx < 0 or ket_idx >= int(drt_ket.ncsf):
        raise IndexError("ket_idx out of range")

    if int(drt_bra.norb) != int(drt_ket.norb):
        raise ValueError("drt_bra and drt_ket must have the same norb")
    if int(drt_bra.nelec) != int(drt_ket.nelec):
        raise ValueError("drt_bra and drt_ket must have the same nelec")

    if not _tri_ok_twos(int(drt_bra.twos_target), 2, int(drt_ket.twos_target)):
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    if cache_ket is None:
        cache_ket = _get_triplet_action_cache(drt_ket)
    if cache_bra is None:
        cache_bra = _get_triplet_action_cache(drt_bra)

    if drt_bra is drt_ket:
        # Candidate bra indices from the scalar E_pq connectivity (same DRT).
        steps = cache_ket.steps[ket_idx]
        occ_p = int(_STEP_TO_OCC[int(steps[p])])
        occ_q = int(_STEP_TO_OCC[int(steps[q])])
        if occ_q <= 0 or occ_p >= 2:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

        epq_cache = _get_epq_action_cache(drt_ket)
        csr = _csr_for_epq(epq_cache, drt_ket, p, q)
        start = int(csr.indptr[ket_idx])
        end = int(csr.indptr[ket_idx + 1])
        idx = csr.indices[start:end].astype(np.int32, copy=False)
        if idx.size == 0:
            return idx, np.zeros(0, dtype=np.float64)

        node_twos = np.asarray(drt_ket.node_twos, dtype=np.int32, order="C")
        ket_steps = cache_ket.steps[ket_idx]
        ket_twos = node_twos[cache_ket.nodes[ket_idx]]

        coeff = np.empty(int(idx.size), dtype=np.float64)
        for k, bra_idx in enumerate(idx.tolist()):
            bra_steps = cache_bra.steps[int(bra_idx)]
            bra_twos = node_twos[cache_bra.nodes[int(bra_idx)]]
            coeff[k] = rme_triplet_single_excitation(
                bra_steps=bra_steps,
                ket_steps=ket_steps,
                bra_twos_prefix=bra_twos,
                ket_twos_prefix=ket_twos,
                twos_bra_total=int(drt_ket.twos_target),
                twos_ket_total=int(drt_ket.twos_target),
                p=p,
                q=q,
            )
        nz = np.nonzero(coeff != 0.0)[0]
        return idx[nz], coeff[nz]

    # Cross-DRT DFS enumerator (correctness-first).
    ket_steps = cache_ket.steps[ket_idx].astype(np.int8, copy=False)
    occ_ket = _STEP_TO_OCC[ket_steps].astype(np.int8, copy=False)
    occ_p = int(occ_ket[p])
    occ_q = int(occ_ket[q])
    if occ_q <= 0 or occ_p >= 2:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    occ_target = occ_ket.astype(np.int8, copy=True)
    occ_target[p] = np.int8(occ_p + 1)
    occ_target[q] = np.int8(occ_q - 1)
    if int(occ_target[p]) > 2 or int(occ_target[q]) < 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    norb = int(drt_ket.norb)
    lo = min(p, q)
    hi = max(p, q)

    child = np.asarray(drt_bra.child, dtype=np.int32, order="C")
    prefix = _child_prefix_walks(drt_bra)
    root = int(drt_bra.root)
    leaf = int(drt_bra.leaf)

    bra_steps = ket_steps.copy()
    nodes = np.empty(norb + 1, dtype=np.int32)
    nodes[0] = root

    # Fixed prefix: enforce bra==ket outside interval.
    idx0 = 0
    node0 = root
    for t in range(lo):
        step = int(bra_steps[t])
        node1 = int(child[node0, step])
        if node1 < 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)
        idx0 += int(prefix[node0, step])
        node0 = node1
        nodes[t + 1] = node0

    node_twos_bra = np.asarray(drt_bra.node_twos, dtype=np.int32, order="C")
    node_twos_ket = np.asarray(drt_ket.node_twos, dtype=np.int32, order="C")
    ket_twos = node_twos_ket[cache_ket.nodes[ket_idx]]

    out_idx: list[int] = []
    out_coeff: list[float] = []

    def dfs(t: int, node: int, idx: int) -> None:
        if t == hi + 1:
            # Fixed suffix: enforce bra==ket outside interval.
            node_s = int(node)
            idx_s = int(idx)
            for u in range(hi + 1, norb):
                step_u = int(bra_steps[u])
                node1 = int(child[node_s, step_u])
                if node1 < 0:
                    return
                idx_s += int(prefix[node_s, step_u])
                node_s = node1
                nodes[u + 1] = node_s
            if node_s != leaf:
                return

            bra_twos = node_twos_bra[nodes]
            cij = rme_triplet_single_excitation(
                bra_steps=bra_steps,
                ket_steps=ket_steps,
                bra_twos_prefix=bra_twos,
                ket_twos_prefix=ket_twos,
                twos_bra_total=int(drt_bra.twos_target),
                twos_ket_total=int(drt_ket.twos_target),
                p=p,
                q=q,
            )
            if cij != 0.0:
                out_idx.append(int(idx_s))
                out_coeff.append(float(cij))
            return

        occ_t = int(occ_target[t])
        if occ_t == 0:
            steps = (0,)
        elif occ_t == 2:
            steps = (3,)
        else:
            steps = (1, 2)  # SOMO: U/L

        for step in steps:
            node1 = int(child[int(node), int(step)])
            if node1 < 0:
                continue
            bra_steps[t] = np.int8(step)
            nodes[t + 1] = node1
            dfs(t + 1, node1, int(idx) + int(prefix[int(node), int(step)]))

    dfs(lo, node0, idx0)

    if not out_idx:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)
    return np.asarray(out_idx, dtype=np.int32), np.asarray(out_coeff, dtype=np.float64)


def apply_triplet_pq(
    drt_bra: DRT,
    drt_ket: DRT,
    c_ket: np.ndarray,
    p: int,
    q: int,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute `out = (T^(1)_{p q}) |ket>` in the bra CSF basis (reduced operator).

    Same-DRT uses the cached scalar `E_pq` connectivity for indices.

    Cross-DRT uses `triplet_contribs_one`'s DFS enumerator and is intended for
    small correctness/validation runs.
    """

    p = int(p)
    q = int(q)
    c = np.asarray(c_ket, dtype=np.float64).ravel()
    if int(c.size) != int(drt_ket.ncsf):
        raise ValueError("c_ket has wrong length for drt_ket")

    if out is None:
        out = np.zeros(int(drt_bra.ncsf), dtype=np.float64)
    else:
        out = np.asarray(out, dtype=np.float64).ravel()
        if int(out.size) != int(drt_bra.ncsf):
            raise ValueError("out has wrong length for drt_bra")
        out.fill(0.0)

    if int(drt_bra.norb) != int(drt_ket.norb):
        raise ValueError("drt_bra and drt_ket must have the same norb")
    if int(drt_bra.nelec) != int(drt_ket.nelec):
        raise ValueError("drt_bra and drt_ket must have the same nelec")

    # Total-spin selection rule for rank-1 (covers singlet->singlet and |ΔS|>1).
    if not _tri_ok_twos(int(drt_bra.twos_target), 2, int(drt_ket.twos_target)):
        return out

    if p == q:
        # TODO: diagonal (p==q) triplet generator (spin-density operator).
        return out

    cache_ket = _get_triplet_action_cache(drt_ket)
    cache_bra = _get_triplet_action_cache(drt_bra) if drt_bra is not drt_ket else cache_ket

    # Same-DRT fast path.
    if drt_bra is drt_ket:
        drt = drt_ket
        ncsf = int(drt.ncsf)
        cache = cache_ket
        epq_cache = _get_epq_action_cache(drt)
        csr = _csr_for_epq(epq_cache, drt, p, q)
        node_twos = np.asarray(drt.node_twos, dtype=np.int32, order="C")

        for ket_idx in range(ncsf):
            amp = float(c[ket_idx])
            if amp == 0.0:
                continue
            start = int(csr.indptr[ket_idx])
            end = int(csr.indptr[ket_idx + 1])
            if start == end:
                continue

            ket_steps = cache.steps[ket_idx]
            ket_twos = node_twos[cache.nodes[ket_idx]]

            for bra_idx in csr.indices[start:end].tolist():
                bra_idx_i = int(bra_idx)
                bra_steps = cache.steps[bra_idx_i]
                bra_twos = node_twos[cache.nodes[bra_idx_i]]
                cij = rme_triplet_single_excitation(
                    bra_steps=bra_steps,
                    ket_steps=ket_steps,
                    bra_twos_prefix=bra_twos,
                    ket_twos_prefix=ket_twos,
                    twos_bra_total=int(drt.twos_target),
                    twos_ket_total=int(drt.twos_target),
                    p=p,
                    q=q,
                )
                if cij != 0.0:
                    out[bra_idx_i] += float(cij) * amp
        return out

    # Cross-DRT slow path: enumerate contributions per ket CSF.
    for ket_idx in range(int(drt_ket.ncsf)):
        amp = float(c[ket_idx])
        if amp == 0.0:
            continue
        bra_idx, coeff = triplet_contribs_one(
            drt_bra,
            drt_ket,
            ket_idx,
            p,
            q,
            cache_bra=cache_bra,
            cache_ket=cache_ket,
        )
        if bra_idx.size:
            out[bra_idx] += coeff * amp
    return out
