from __future__ import annotations

import numpy as np

from asuka.integrals.df_integrals import DFMOIntegrals
from asuka.cuguga.drt import DRT
from asuka.cuguga.oracle import (
    _STEP_TO_OCC,
    _csr_for_epq,
    _epq_contribs_cached,
    _get_epq_action_cache,
    occ_table,
)


def connected_row_df(drt: DRT, h1e, df_eri: DFMOIntegrals, j: int, max_out: int = 200_000):
    """Return (i_idx, hij) for CSF j using DF/Cholesky-vector ERIs.

    This mirrors :func:`asuka.cuguga.oracle.connected_row`, but replaces
    all dense 4-index ERI usage with a DF representation:

      (pq|rs) ~= sum_L d[L,pq] d[L,rs]

    Notes
    -----
    - This is a prototype path intended for profiling/validation.
    - Numerical results match `connected_row(drt, h1e, eri4, j)` when `eri4` is
      replaced by the reconstructed DF tensor `df_eri.to_eri4()` (up to FP roundoff).
    - DF is an approximation to the *true* ERIs; compare to exact ERIs only with
      an appropriate tolerance.
    """

    j = int(j)
    max_out = int(max_out)
    if max_out < 1:
        raise ValueError("max_out must be >= 1")

    norb = int(drt.norb)
    if int(df_eri.norb) != norb:
        raise ValueError(f"df_eri.norb={int(df_eri.norb)} does not match drt.norb={norb}")

    h1e = np.asarray(h1e, dtype=np.float64)
    if h1e.shape != (norb, norb):
        raise ValueError("h1e has wrong shape")

    cache = _get_epq_action_cache(drt)
    steps = cache.steps[int(j)]
    occ_j = _STEP_TO_OCC[steps].astype(np.int8, copy=False)

    # Spin-free Hamiltonian in generator form:
    #   H = Σ_pq h_pq E_pq + 1/2 Σ_pqrs (pq|rs) (E_pq E_rs - δ_qr E_ps)
    #
    # This row-oracle prototype folds:
    #   - the contraction term (-δ_qr E_ps) into h_eff via J_ps = Σ_q (p q| q s),
    #   - the r==s slice of the 2-body product into h_eff via Σ_r (p q| r r) occ_r(j),
    # to avoid explicitly iterating r==s cases in the 2-body loop.
    h_eff = h1e - 0.5 * np.asarray(df_eri.j_ps, dtype=np.float64)
    h_eff = h_eff + df_eri.rr_slice_h_eff(occ_j, half=0.5)

    ncsf = int(drt.ncsf)
    acc = np.zeros(ncsf, dtype=np.float64)
    acc[int(j)] = float(np.dot(np.diag(h_eff), occ_j.astype(np.float64)))

    # 1-body off-diagonal (p!=q) contributions with effective h_eff.
    src1 = np.nonzero(occ_j > 0)[0].tolist()
    dst1 = np.nonzero(occ_j < 2)[0].tolist()
    for q in src1:
        for p in dst1:
            if p == q:
                continue
            hpq = float(h_eff[p, q])
            if hpq == 0.0:
                continue
            i_idx, coeff = _epq_contribs_cached(cache, drt, j, int(p), int(q))
            if i_idx.size:
                acc[i_idx] += hpq * coeff

    # 2-body product terms with r!=s (r==s already absorbed into h_eff above).
    occ_all = occ_table(drt)

    # Collect all nonzero E_rs|j> contributions (r!=s) grouped by resulting CSF index k.
    by_k: dict[int, list[tuple[int, float]]] = {}
    for s in src1:
        for r in dst1:
            if r == s:
                continue
            csr_rs = _csr_for_epq(cache, drt, int(r), int(s))
            start_rs = int(csr_rs.indptr[j])
            end_rs = int(csr_rs.indptr[j + 1])
            if start_rs == end_rs:
                continue
            rs_id = int(r) * norb + int(s)
            k_idx = csr_rs.indices[start_rs:end_rs]
            coeff_rs = csr_rs.data[start_rs:end_rs]
            for kk, c_rs in zip(k_idx.tolist(), coeff_rs.tolist()):
                by_k.setdefault(int(kk), []).append((rs_id, float(c_rs)))

    # For each intermediate k: build g_pq = Σ_{rs} (1/2)(pq|rs) <k|E_rs|j> via DF, then
    # apply it as a one-body operator Σ_{pq} g_pq E_pq |k>.
    for k, rs_terms in by_k.items():
        rs_ids = np.asarray([t[0] for t in rs_terms], dtype=np.int32)
        rs_coeff = np.asarray([t[1] for t in rs_terms], dtype=np.float64)

        g_flat = df_eri.contract_cols(rs_ids, rs_coeff, half=0.5)
        g = g_flat.reshape(norb, norb)

        occ_k = occ_all[int(k)].astype(np.float64, copy=False)
        acc[int(k)] += float(np.dot(np.diag(g), occ_k))

        src_k = np.nonzero(occ_all[int(k)] > 0)[0].tolist()
        dst_k = np.nonzero(occ_all[int(k)] < 2)[0].tolist()
        for q in src_k:
            for p in dst_k:
                if p == q:
                    continue
                gpq = float(g[p, q])
                if gpq == 0.0:
                    continue
                i_idx, coeff = _epq_contribs_cached(cache, drt, int(k), int(p), int(q))
                if i_idx.size:
                    acc[i_idx] += gpq * coeff

    # Materialize sparse row output.
    nz = np.nonzero(acc)[0].astype(np.int32, copy=False)
    if nz.size:
        others = nz[nz != int(j)]
    else:
        others = np.zeros(0, dtype=np.int32)

    i_idx_arr = np.concatenate(([np.int32(j)], others)).astype(np.int32, copy=False)
    hij_arr = np.concatenate(([np.float64(acc[int(j)])], acc[others])).astype(np.float64, copy=False)

    if i_idx_arr.size > max_out:
        raise ValueError(f"oracle produced {i_idx_arr.size} entries > max_out={max_out}")
    return i_idx_arr, hij_arr
