from __future__ import annotations

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.state_cache import DRTStateCache
from asuka.cuguga.epq.action import path_nodes
from .epq_sample import sample_epq_one

try:
    from asuka.integrals.df_integrals import DFMOIntegrals
except Exception:  # pragma: no cover
    DFMOIntegrals = None  # type: ignore[assignment]

try:  # optional compiled QMC spawn backend (OpenMP-capable)
    from asuka._epq_cy import qmc_spawn_hamiltonian_events_cy as _qmc_spawn_hamiltonian_events_cy  # type: ignore
except Exception:  # pragma: no cover
    _qmc_spawn_hamiltonian_events_cy = None

_STEP_TO_OCC = np.asarray([0, 1, 1, 2], dtype=np.int8)


def spawn_one_body_events(
    drt: DRT,
    h_eff: np.ndarray,
    x_idx: np.ndarray,
    x_val: np.ndarray,
    *,
    eps: float,
    nspawn: int,
    rng: np.random.Generator,
    initiator_t: float = 0.0,
    state_cache: DRTStateCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Spawn events for the one-body operator sum_{pq} h_eff[p,q] E_pq.

    Returns COO arrays `(evt_idx, evt_val)` representing an unbiased stochastic
    estimator of `-eps * H1 * x`, where `x` is a sparse vector over CSF indices.

    Notes
    -----
    - This is a CPU reference implementation intended for early validation.
    - Proposal (first version):
        - choose q uniformly from occupied orbitals (occ[q] > 0),
        - choose p uniformly from all orbitals (0..norb-1),
        - sample one child from E_pq|j> with p(i) ∝ |c_i|.
    - Each parent emits exactly `nspawn` attempted events; each accepted event is
      scaled by `1/nspawn` so the total expectation matches the full sum.
    - Initiator gating is optional; set `initiator_t=0` to disable.
    """

    eps = float(eps)
    nspawn = int(nspawn)
    if nspawn <= 0:
        raise ValueError("nspawn must be > 0")

    x_idx_i32 = np.asarray(x_idx, dtype=np.int32).ravel()
    x_val_f64 = np.asarray(x_val, dtype=np.float64).ravel()
    if x_idx_i32.size != x_val_f64.size:
        raise ValueError("x_idx and x_val must have the same size")

    norb = int(drt.norb)
    h_eff = np.asarray(h_eff, dtype=np.float64).reshape(norb, norb)

    m = int(x_idx_i32.size)
    if m == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    # Worst-case: every attempted event yields one output.
    evt_idx = np.empty(m * nspawn, dtype=np.int32)
    evt_val = np.empty(m * nspawn, dtype=np.float64)
    out = 0

    scale = -eps / float(nspawn)
    t = float(initiator_t)

    for parent_pos in range(m):
        j = int(x_idx_i32[parent_pos])
        xj = float(x_val_f64[parent_pos])
        if xj == 0.0:
            continue

        if state_cache is None:
            steps = drt.index_to_path(j)
            nodes = path_nodes(drt, steps)
        else:
            steps = state_cache.steps[j]
            nodes = state_cache.nodes[j]
        occ = _STEP_TO_OCC[np.asarray(steps, dtype=np.int8, order="C")]

        src = np.nonzero(occ > 0)[0]
        if src.size == 0:
            continue
        inv_p_pair = float(src.size * norb)  # 1 / (1/src.size * 1/norb)

        # Initiator gating (if enabled): allow new targets only if |xj| >= t.
        # The "present set" is the current x support (x_idx_i32 is assumed sorted unique).
        allow_new = True
        if t > 0.0 and abs(xj) < t:
            allow_new = False

        for _ in range(nspawn):
            q = int(src[int(rng.integers(0, int(src.size)))])
            p = int(rng.integers(0, norb))

            w = float(h_eff[p, q])
            if w == 0.0:
                continue

            s = sample_epq_one(drt, j, p, q, rng, steps=steps, nodes=nodes)
            if not s.valid:
                continue

            i = int(s.child)
            if not allow_new:
                k = int(np.searchsorted(x_idx_i32, i))
                if k >= m or int(x_idx_i32[k]) != i:
                    continue

            evt_idx[out] = i
            evt_val[out] = scale * xj * w * float(s.coeff) * float(s.inv_p) * inv_p_pair
            out += 1

    return evt_idx[:out], evt_val[:out]


def _as_eri_mat(eri: np.ndarray, *, norb: int) -> np.ndarray:
    eri = np.asarray(eri, dtype=np.float64)
    nops = int(norb) * int(norb)
    if eri.ndim == 2:
        if eri.shape != (nops, nops):
            raise ValueError(f"eri_mat has wrong shape: {eri.shape} (expected {(nops, nops)})")
        return eri
    if eri.ndim == 4:
        if eri.shape != (int(norb), int(norb), int(norb), int(norb)):
            raise ValueError("eri4 has wrong shape")
        return eri.reshape(nops, nops)
    raise ValueError("eri must be eri_mat[pq,rs] (2D) or eri4[p,q,r,s] (4D)")


def _as_eri4(eri: np.ndarray, *, norb: int) -> np.ndarray:
    eri = np.asarray(eri, dtype=np.float64)
    if eri.ndim == 4:
        if eri.shape != (int(norb), int(norb), int(norb), int(norb)):
            raise ValueError("eri4 has wrong shape")
        return eri
    if eri.ndim == 2:
        nops = int(norb) * int(norb)
        if eri.shape != (nops, nops):
            raise ValueError(f"eri_mat has wrong shape: {eri.shape} (expected {(nops, nops)})")
        return eri.reshape(int(norb), int(norb), int(norb), int(norb))
    raise ValueError("eri must be eri4[p,q,r,s] (4D) or eri_mat[pq,rs] (2D)")


def _is_df_eri(eri) -> bool:
    return DFMOIntegrals is not None and isinstance(eri, DFMOIntegrals)


def spawn_two_body_events(
    drt: DRT,
    eri,
    x_idx: np.ndarray,
    x_val: np.ndarray,
    *,
    eps: float,
    nspawn: int,
    rng: np.random.Generator,
    initiator_t: float = 0.0,
    state_cache: DRTStateCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Spawn events for the 2-body product term: (1/2) Σ_{pqrs} (pq|rs) E_pq E_rs.

    Returns COO arrays `(evt_idx, evt_val)` representing an unbiased stochastic
    estimator of `-eps * H2 * x`, where `x` is a sparse vector over CSF indices.

    Proposal (first version, correctness-focused):
    - For the first generator (E_rs|j>):
        - choose s uniformly from occupied orbitals of j (occ[s] > 0),
        - choose r uniformly from all orbitals (0..norb-1).
    - For the second generator (E_pq|k>):
        - choose q uniformly from occupied orbitals of k (occ_k[q] > 0),
        - choose p uniformly from all orbitals (0..norb-1).

    This covers diagonal generators (r==s, p==q) naturally via `sample_epq_one`.
    """

    eps = float(eps)
    nspawn = int(nspawn)
    if nspawn <= 0:
        raise ValueError("nspawn must be > 0")

    x_idx_i32 = np.asarray(x_idx, dtype=np.int32).ravel()
    x_val_f64 = np.asarray(x_val, dtype=np.float64).ravel()
    if x_idx_i32.size != x_val_f64.size:
        raise ValueError("x_idx and x_val must have the same size")

    norb = int(drt.norb)
    use_df = _is_df_eri(eri)
    if use_df:
        df_eri = eri
        if int(df_eri.norb) != norb:
            raise ValueError(f"df_eri.norb={int(df_eri.norb)} does not match drt.norb={norb}")
        eri_mat = None
    else:
        df_eri = None
        eri_mat = _as_eri_mat(eri, norb=norb)

    m = int(x_idx_i32.size)
    if m == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    evt_idx = np.empty(m * nspawn, dtype=np.int32)
    evt_val = np.empty(m * nspawn, dtype=np.float64)
    out = 0

    # For DF, defer evaluating v_pqrs until after acceptance and do it in a single
    # vectorized pass.
    if use_df:
        pq_ids = np.empty(m * nspawn, dtype=np.int32)
        rs_ids = np.empty(m * nspawn, dtype=np.int32)

    scale = -eps / float(nspawn)
    t = float(initiator_t)

    for parent_pos in range(m):
        j = int(x_idx_i32[parent_pos])
        xj = float(x_val_f64[parent_pos])
        if xj == 0.0:
            continue

        if state_cache is None:
            steps_j = drt.index_to_path(j)
            nodes_j = path_nodes(drt, steps_j)
        else:
            steps_j = state_cache.steps[j]
            nodes_j = state_cache.nodes[j]
        occ_j = _STEP_TO_OCC[np.asarray(steps_j, dtype=np.int8, order="C")]

        src_j = np.nonzero(occ_j > 0)[0]
        if src_j.size == 0:
            continue
        inv_p_rs = float(src_j.size * norb)  # 1 / (1/src_j.size * 1/norb)

        allow_new = True
        if t > 0.0 and abs(xj) < t:
            allow_new = False

        for _ in range(nspawn):
            s = int(src_j[int(rng.integers(0, int(src_j.size)))])
            r = int(rng.integers(0, norb))
            rs_id = int(r) * norb + int(s)

            samp_rs = sample_epq_one(drt, j, r, s, rng, steps=steps_j, nodes=nodes_j)
            if not samp_rs.valid:
                continue

            k = int(samp_rs.child)

            if state_cache is None:
                steps_k = drt.index_to_path(k)
                nodes_k = path_nodes(drt, steps_k)
            else:
                steps_k = state_cache.steps[k]
                nodes_k = state_cache.nodes[k]
            occ_k = _STEP_TO_OCC[np.asarray(steps_k, dtype=np.int8, order="C")]
            src_k = np.nonzero(occ_k > 0)[0]
            if src_k.size == 0:
                continue
            inv_p_pq = float(src_k.size * norb)

            q = int(src_k[int(rng.integers(0, int(src_k.size)))])
            p = int(rng.integers(0, norb))
            pq_id = int(p) * norb + int(q)

            samp_pq = sample_epq_one(drt, k, p, q, rng, steps=steps_k, nodes=nodes_k)
            if not samp_pq.valid:
                continue

            i = int(samp_pq.child)
            if not allow_new:
                pos = int(np.searchsorted(x_idx_i32, i))
                if pos >= m or int(x_idx_i32[pos]) != i:
                    continue

            evt_idx[out] = i
            pref = (
                scale
                * xj
                * float(samp_rs.coeff)
                * float(samp_rs.inv_p)
                * inv_p_rs
                * float(samp_pq.coeff)
                * float(samp_pq.inv_p)
                * inv_p_pq
            )
            if use_df:
                pq_ids[out] = np.int32(pq_id)
                rs_ids[out] = np.int32(rs_id)
                evt_val[out] = pref
            else:
                v_pqrs = 0.5 * float(eri_mat[pq_id, rs_id])
                if v_pqrs == 0.0:
                    continue
                evt_val[out] = pref * v_pqrs
            out += 1

    if out == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    if use_df:
        l_full = np.asarray(df_eri.l_full, dtype=np.float64)
        v = 0.5 * np.einsum("ij,ij->i", l_full[pq_ids[:out]], l_full[rs_ids[:out]], optimize=True)
        evt_val[:out] *= v

    keep = evt_val[:out] != 0.0
    return evt_idx[:out][keep], evt_val[:out][keep]


def spawn_hamiltonian_events(
    drt: DRT,
    h1e: np.ndarray,
    eri,
    x_idx: np.ndarray,
    x_val: np.ndarray,
    *,
    eps: float,
    nspawn_one: int,
    nspawn_two: int,
    rng: np.random.Generator,
    initiator_t: float = 0.0,
    state_cache: DRTStateCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Spawn events for the full spin-free Hamiltonian (CPU reference path).

    Uses the same decomposition as the deterministic sparse-row oracle:

      H = Σ_pq h_pq E_pq + 1/2 Σ_pqrs (pq|rs) (E_pq E_rs - δ_qr E_ps)

    Approach:
    - Fold the contraction term and r==s slice into a state-dependent effective
      one-body operator h_eff(j) using the CSF occupation vector occ(j).
    - Sample only the r!=s two-body product term via sequential generator
      sampling E_rs then E_pq.

    Returns COO arrays `(evt_idx, evt_val)` representing an unbiased stochastic
    estimator of `-eps * H * x`.
    """

    eps = float(eps)
    nspawn_one = int(nspawn_one)
    nspawn_two = int(nspawn_two)
    if nspawn_one < 0 or nspawn_two < 0:
        raise ValueError("nspawn_one and nspawn_two must be >= 0")
    if nspawn_one == 0 and nspawn_two == 0:
        raise ValueError("at least one of nspawn_one or nspawn_two must be > 0")

    x_idx_i32 = np.asarray(x_idx, dtype=np.int32).ravel()
    x_val_f64 = np.asarray(x_val, dtype=np.float64).ravel()
    if x_idx_i32.size != x_val_f64.size:
        raise ValueError("x_idx and x_val must have the same size")

    norb = int(drt.norb)
    h1e = np.asarray(h1e, dtype=np.float64).reshape(norb, norb)

    use_df = _is_df_eri(eri)
    if use_df:
        df_eri = eri
        if int(df_eri.norb) != norb:
            raise ValueError(f"df_eri.norb={int(df_eri.norb)} does not match drt.norb={norb}")
    else:
        df_eri = None

    # Compiled backend: requires a state_cache table (steps/nodes) and uses eri_mat directly.
    if (not use_df) and state_cache is not None and _qmc_spawn_hamiltonian_events_cy is not None:
        eri_mat = _as_eri_mat(eri, norb=norb)
        seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        evt_i_full, evt_v_full = _qmc_spawn_hamiltonian_events_cy(
            drt,
            h1e,
            eri_mat,
            state_cache.steps,
            state_cache.nodes,
            x_idx_i32,
            x_val_f64,
            float(eps),
            int(nspawn_one),
            int(nspawn_two),
            np.uint64(seed),
            float(initiator_t),
        )
        keep = evt_i_full >= 0
        return evt_i_full[keep], evt_v_full[keep]

    if use_df:
        # h_base[p,q] = h1e[p,q] - 0.5 * J_ps[p,q]
        h_base = h1e - 0.5 * np.asarray(df_eri.j_ps, dtype=np.float64)
        eri_mat = None
    else:
        eri4 = _as_eri4(eri, norb=norb)
        eri_mat = eri4.reshape(norb * norb, norb * norb)
        # h_base[p,q] = h1e[p,q] - 0.5 * Σ_t (p t | t q)
        h_base = h1e - 0.5 * np.einsum("pqqs->ps", eri4, optimize=True)
        eri_pq_rr = np.diagonal(eri4, axis1=2, axis2=3)  # (p,q,r) -> (p q | r r)

    m = int(x_idx_i32.size)
    if m == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    nspawn_one_eff = int(nspawn_one)
    nspawn_two_eff = int(nspawn_two)
    nspawn_total = nspawn_one_eff + nspawn_two_eff
    evt_idx = np.empty(m * nspawn_total, dtype=np.int32)
    evt_val = np.empty(m * nspawn_total, dtype=np.float64)
    out = 0

    if use_df and nspawn_two_eff:
        # Track only accepted two-body events for vectorized DF dot products.
        two_pos = np.empty(m * nspawn_two_eff, dtype=np.int32)
        two_pq = np.empty(m * nspawn_two_eff, dtype=np.int32)
        two_rs = np.empty(m * nspawn_two_eff, dtype=np.int32)
        two_out = 0

    scale_one = (-eps / float(nspawn_one_eff)) if nspawn_one_eff else 0.0
    scale_two = (-eps / float(nspawn_two_eff)) if nspawn_two_eff else 0.0
    t = float(initiator_t)

    for parent_pos in range(m):
        j = int(x_idx_i32[parent_pos])
        xj = float(x_val_f64[parent_pos])
        if xj == 0.0:
            continue

        if state_cache is None:
            steps_j = drt.index_to_path(j)
            nodes_j = path_nodes(drt, steps_j)
        else:
            steps_j = state_cache.steps[j]
            nodes_j = state_cache.nodes[j]
        occ_j = _STEP_TO_OCC[np.asarray(steps_j, dtype=np.int8, order="C")]

        # h_eff(j)[p,q] = h_base[p,q] + 0.5 * Σ_r (p q | r r) occ_j[r]
        if use_df:
            h_eff_j = h_base + df_eri.rr_slice_h_eff(occ_j, half=0.5)
        else:
            h_eff_j = h_base + 0.5 * np.tensordot(eri_pq_rr, occ_j, axes=(2, 0))

        src_j = np.nonzero(occ_j > 0)[0]
        if src_j.size == 0:
            continue

        allow_new = True
        if t > 0.0 and abs(xj) < t:
            allow_new = False

        # One-body part: sample q from occupied orbitals, p uniform over all orbitals.
        if nspawn_one_eff:
            inv_p_pair_one = float(src_j.size * norb)
            for _ in range(nspawn_one_eff):
                q = int(src_j[int(rng.integers(0, int(src_j.size)))])
                p = int(rng.integers(0, norb))

                w = float(h_eff_j[p, q])
                if w == 0.0:
                    continue

                s = sample_epq_one(drt, j, p, q, rng, steps=steps_j, nodes=nodes_j)
                if not s.valid:
                    continue

                i = int(s.child)
                if not allow_new:
                    k = int(np.searchsorted(x_idx_i32, i))
                    if k >= m or int(x_idx_i32[k]) != i:
                        continue

                evt_idx[out] = i
                evt_val[out] = scale_one * xj * w * float(s.coeff) * float(s.inv_p) * inv_p_pair_one
                out += 1

        # Two-body r!=s product part: sample s from occupied orbitals, r uniform over all orbitals excluding s.
        if nspawn_two_eff:
            if norb <= 1:
                continue
            dst_j = np.nonzero(occ_j < 2)[0]
            for _ in range(nspawn_two_eff):
                s_orb = int(src_j[int(rng.integers(0, int(src_j.size)))])

                dst_size = int(dst_j.size)
                if dst_size == 0:
                    continue
                occ_s = int(occ_j[s_orb])
                if occ_s == 1:
                    # Exclude r==s (s is in dst_j in this case).
                    if dst_size <= 1:
                        continue
                    pos_s = int(np.searchsorted(dst_j, s_orb))
                    u = int(rng.integers(0, dst_size - 1))
                    if u >= pos_s:
                        u += 1
                    r_orb = int(dst_j[u])
                    inv_p_rs = float(src_j.size * (dst_size - 1))
                else:
                    # occ_s == 2: s not in dst_j.
                    u = int(rng.integers(0, dst_size))
                    r_orb = int(dst_j[u])
                    inv_p_rs = float(src_j.size * dst_size)
                rs_id = int(r_orb) * norb + int(s_orb)

                samp_rs = sample_epq_one(drt, j, r_orb, s_orb, rng, steps=steps_j, nodes=nodes_j)
                if not samp_rs.valid:
                    continue
                k_csf = int(samp_rs.child)

                if state_cache is None:
                    steps_k = drt.index_to_path(k_csf)
                    nodes_k = path_nodes(drt, steps_k)
                else:
                    steps_k = state_cache.steps[k_csf]
                    nodes_k = state_cache.nodes[k_csf]
                occ_k = _STEP_TO_OCC[np.asarray(steps_k, dtype=np.int8, order="C")]
                src_k = np.nonzero(occ_k > 0)[0]
                if src_k.size == 0:
                    continue
                inv_p_pq = float(src_k.size * norb)

                q_orb = int(src_k[int(rng.integers(0, int(src_k.size)))])
                p_orb = int(rng.integers(0, norb))
                pq_id = int(p_orb) * norb + int(q_orb)

                samp_pq = sample_epq_one(drt, k_csf, p_orb, q_orb, rng, steps=steps_k, nodes=nodes_k)
                if not samp_pq.valid:
                    continue

                i = int(samp_pq.child)
                if not allow_new:
                    pos = int(np.searchsorted(x_idx_i32, i))
                    if pos >= m or int(x_idx_i32[pos]) != i:
                        continue

                evt_idx[out] = i
                pref = (
                    scale_two
                    * xj
                    * float(samp_rs.coeff)
                    * float(samp_rs.inv_p)
                    * inv_p_rs
                    * float(samp_pq.coeff)
                    * float(samp_pq.inv_p)
                    * inv_p_pq
                )
                if use_df:
                    two_pos[two_out] = np.int32(out)
                    two_pq[two_out] = np.int32(pq_id)
                    two_rs[two_out] = np.int32(rs_id)
                    evt_val[out] = pref
                    two_out += 1
                else:
                    v_pqrs = 0.5 * float(eri_mat[pq_id, rs_id])
                    if v_pqrs == 0.0:
                        continue
                    evt_val[out] = pref * v_pqrs
                out += 1

    if out == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    if use_df and nspawn_two_eff and two_out:
        l_full = np.asarray(df_eri.l_full, dtype=np.float64)
        v = 0.5 * np.einsum("ij,ij->i", l_full[two_pq[:two_out]], l_full[two_rs[:two_out]], optimize=True)
        evt_val[two_pos[:two_out]] *= v

    keep = evt_val[:out] != 0.0
    return evt_idx[:out][keep], evt_val[:out][keep]


def spawn_hamiltonian_events_semi_stochastic(
    drt: DRT,
    h1e: np.ndarray,
    eri,
    x_idx: np.ndarray,
    x_val: np.ndarray,
    *,
    eps: float,
    nspawn_one: int,
    nspawn_two: int,
    rng: np.random.Generator,
    initiator_t: float = 0.0,
    state_cache: DRTStateCache | None = None,
    det_nparent: int = 0,
    det_abs_x_threshold: float | None = None,
    det_max_out: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Semi-stochastic version of :func:`spawn_hamiltonian_events`.

    Deterministically adds the *exact* `-eps * H * x_j` contributions for a small
    subset of parents `j` (picked by `|x_j|`), and uses the standard stochastic
    spawner for the remaining parents.

    This reduces variance in the spawned estimator while remaining unbiased
    (ignoring initiator gating which is itself a controlled bias).

    Parameters
    ----------
    det_nparent
        Number of parents treated deterministically (top-|x_j|). If <=0, no
        deterministic parents are selected unless `det_abs_x_threshold` selects
        some.
    det_abs_x_threshold
        Optional absolute amplitude cutoff: include any parent with
        `|x_j| >= det_abs_x_threshold` in the deterministic set (in addition to
        `det_nparent`).
    det_max_out
        Only used for dense-ERI connected-row oracle. Defaults to `drt.ncsf`.
    """

    eps = float(eps)
    nspawn_one = int(nspawn_one)
    nspawn_two = int(nspawn_two)
    x_idx_i32 = np.asarray(x_idx, dtype=np.int32).ravel()
    x_val_f64 = np.asarray(x_val, dtype=np.float64).ravel()
    if x_idx_i32.size != x_val_f64.size:
        raise ValueError("x_idx and x_val must have the same size")
    if x_idx_i32.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    det_nparent = int(det_nparent)
    if det_nparent < 0:
        raise ValueError("det_nparent must be >= 0")
    if det_abs_x_threshold is not None and not np.isfinite(float(det_abs_x_threshold)):
        raise ValueError("det_abs_x_threshold must be finite when provided")

    m = int(x_idx_i32.size)
    abs_x = np.abs(x_val_f64)

    det_mask = np.zeros(m, dtype=np.bool_)
    if det_abs_x_threshold is not None:
        det_mask |= abs_x >= float(det_abs_x_threshold)
    if det_nparent > 0:
        k = min(int(det_nparent), m)
        pos = np.argpartition(abs_x, -k)[-k:]
        det_mask[pos] = True

    if not np.any(det_mask):
        if nspawn_one < 0 or nspawn_two < 0:
            raise ValueError("nspawn_one and nspawn_two must be >= 0")
        return spawn_hamiltonian_events(
            drt,
            h1e,
            eri,
            x_idx_i32,
            x_val_f64,
            eps=float(eps),
            nspawn_one=int(nspawn_one),
            nspawn_two=int(nspawn_two),
            rng=rng,
            initiator_t=float(initiator_t),
            state_cache=state_cache,
        )

    # Deterministic parents.
    use_df = _is_df_eri(eri)
    if use_df:
        from asuka.cuguga.oracle.sparse import connected_row_sparse_df  # noqa: PLC0415

        df_eri = eri
    else:
        from asuka.cuguga.oracle.sparse import connected_row_sparse  # noqa: PLC0415

        eri4 = _as_eri4(eri, norb=int(drt.norb))

    if det_max_out is None:
        det_max_out = int(getattr(drt, "ncsf", 200_000))
    det_max_out = int(det_max_out)

    det_idx_chunks: list[np.ndarray] = []
    det_val_chunks: list[np.ndarray] = []
    t = float(initiator_t)

    # Use the full original support for initiator membership checks.
    x_support = x_idx_i32

    det_pos = np.nonzero(det_mask)[0].astype(np.int64, copy=False)
    for parent_pos in det_pos.tolist():
        j = int(x_idx_i32[int(parent_pos)])
        xj = float(x_val_f64[int(parent_pos)])
        if xj == 0.0:
            continue

        allow_new = True
        if t > 0.0 and abs(xj) < t:
            allow_new = False

        if use_df:
            i_idx, hij = connected_row_sparse_df(drt, np.asarray(h1e, dtype=np.float64), df_eri, j)
        else:
            i_idx, hij = connected_row_sparse(
                drt,
                np.asarray(h1e, dtype=np.float64),
                eri4,
                int(j),
                max_out=int(det_max_out),
                state_cache=state_cache,
            )

        if i_idx.size == 0:
            continue
        i_idx = np.asarray(i_idx, dtype=np.int32, order="C").ravel()
        hij = np.asarray(hij, dtype=np.float64, order="C").ravel()
        if i_idx.size != hij.size:
            raise RuntimeError("connected_row_sparse returned mismatched idx/val sizes")

        if not allow_new:
            # Filter to children already present in x_support (initiator gating).
            pos = np.searchsorted(x_support, i_idx)
            in_range = (pos >= 0) & (pos < m)
            keep = np.zeros_like(in_range)
            if np.any(in_range):
                pos2 = pos[in_range]
                keep[in_range] = x_support[pos2] == i_idx[in_range]
            if not np.any(keep):
                continue
            i_idx = i_idx[keep]
            hij = hij[keep]

        det_idx_chunks.append(i_idx)
        det_val_chunks.append((-float(eps) * float(xj)) * hij)

    # Stochastic parents: exclude deterministic ones.
    rem_mask = ~det_mask
    x_idx_rem = x_idx_i32[rem_mask]
    x_val_rem = x_val_f64[rem_mask]

    st_idx = np.zeros(0, dtype=np.int32)
    st_val = np.zeros(0, dtype=np.float64)
    if x_idx_rem.size:
        if nspawn_one < 0 or nspawn_two < 0:
            raise ValueError("nspawn_one and nspawn_two must be >= 0")
        if nspawn_one == 0 and nspawn_two == 0:
            raise ValueError("at least one of nspawn_one or nspawn_two must be > 0 when stochastic parents exist")
        st_idx, st_val = spawn_hamiltonian_events(
            drt,
            h1e,
            eri,
            x_idx_rem,
            x_val_rem,
            eps=float(eps),
            nspawn_one=int(nspawn_one),
            nspawn_two=int(nspawn_two),
            rng=rng,
            initiator_t=float(initiator_t),
            state_cache=state_cache,
        )

    if det_idx_chunks:
        det_idx = np.concatenate(det_idx_chunks).astype(np.int32, copy=False)
        det_val = np.concatenate(det_val_chunks).astype(np.float64, copy=False)
    else:
        det_idx = np.zeros(0, dtype=np.int32)
        det_val = np.zeros(0, dtype=np.float64)

    if st_idx.size == 0:
        return det_idx, det_val
    if det_idx.size == 0:
        return np.asarray(st_idx, dtype=np.int32, order="C"), np.asarray(st_val, dtype=np.float64, order="C")
    return (
        np.concatenate((det_idx, np.asarray(st_idx, dtype=np.int32, order="C"))).astype(np.int32, copy=False),
        np.concatenate((det_val, np.asarray(st_val, dtype=np.float64, order="C"))).astype(np.float64, copy=False),
    )
