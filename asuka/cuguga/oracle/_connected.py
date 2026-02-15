"""Connected row generation and related utilities.

This module provides functions for computing Hamiltonian matrix rows
(connected CSF indices and coupling coefficients) using the spin-free
generator formalism:

.. math::

    H = \\sum_{pq} h_{pq} E_{pq}
      + \\frac{1}{2} \\sum_{pqrs} (pq|rs)(E_{pq} E_{rs} - \\delta_{qr} E_{ps})

Two row oracles are provided:

- :func:`connected_row`: Full dense-accumulator oracle using global E_pq
  CSR caches (fast for small-to-medium CSF spaces).
- :func:`connected_row_structure_only`: Phase-0 oracle returning candidate
  indices based on occupancy-difference screening (no integral evaluation).
"""

from __future__ import annotations

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.oracle._cache import (
    _STEP_TO_OCC,
    _csr_for_epq,
    _epq_contribs_cached,
    _get_epq_action_cache,
    occ_groups,
    occ_table,
)


# ---------------------------------------------------------------------------
# ERI restoration helper
# ---------------------------------------------------------------------------


def _restore_eri_4d(eri, norb: int) -> np.ndarray:
    """Restore a packed ERI to full 4D ``(norb, norb, norb, norb)`` form.

    Parameters
    ----------
    eri : array_like
        Electron repulsion integrals in packed or full form.
    norb : int
        Number of orbitals.

    Returns
    -------
    np.ndarray
        Shape ``(norb, norb, norb, norb)`` array of ERIs.
    """
    from asuka.cuguga.eri import restore_eri1  # noqa: PLC0415

    return restore_eri1(eri, int(norb))


# ---------------------------------------------------------------------------
# Spin-occupation assignment
# ---------------------------------------------------------------------------


def _assign_spin_occupations(
    occ: np.ndarray, *, neleca: int, nelecb: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(alpha_occ, beta_occ)`` as 0/1 arrays consistent with ``occ``.

    Given a spatial occupation pattern, assign alpha and beta electrons
    to singly-occupied orbitals consistent with the electron counts.

    Parameters
    ----------
    occ : np.ndarray
        Spatial occupation (0/1/2) for each orbital.
    neleca : int
        Number of alpha electrons.
    nelecb : int
        Number of beta electrons.

    Returns
    -------
    alpha : np.ndarray
        int8 array of alpha occupations (0 or 1).
    beta : np.ndarray
        int8 array of beta occupations (0 or 1).
    """

    occ = np.asarray(occ, dtype=np.int8).ravel()
    doubly = np.nonzero(occ == 2)[0]
    single = np.nonzero(occ == 1)[0]

    alpha = np.zeros_like(occ, dtype=np.int8)
    beta = np.zeros_like(occ, dtype=np.int8)
    alpha[doubly] = 1
    beta[doubly] = 1

    ndoubly = int(doubly.size)
    alpha_need = int(neleca) - ndoubly
    beta_need = int(nelecb) - ndoubly
    if alpha_need < 0 or beta_need < 0:
        raise ValueError("invalid (neleca, nelecb) for occupancy pattern")
    if alpha_need + beta_need != int(single.size):
        raise ValueError("invalid (neleca, nelecb) for occupancy pattern")

    alpha[single[:alpha_need]] = 1
    beta[single[alpha_need:]] = 1
    return alpha, beta


# ---------------------------------------------------------------------------
# Diagonal element guess
# ---------------------------------------------------------------------------


def diagonal_element_det_guess(
    drt: DRT,
    h1e,
    eri,
    j: int,
    *,
    eri_ppqq: np.ndarray | None = None,
    eri_pqqp: np.ndarray | None = None,
) -> float:
    """Determinant-based diagonal guess for CSF j.

    Computes the diagonal Hamiltonian element using a single-determinant
    approximation. This is exact for closed-shell CSFs and serves as a
    useful preconditioner for the Davidson solver.

    Parameters
    ----------
    drt : DRT
        The Distinct Row Table.
    h1e : array_like
        One-electron integrals, shape ``(norb, norb)``.
    eri : array_like
        Two-electron integrals (packed or full form).
    j : int
        CSF index.
    eri_ppqq : np.ndarray, optional
        Precomputed Coulomb integrals ``(pp|qq)``.
    eri_pqqp : np.ndarray, optional
        Precomputed exchange integrals ``(pq|qp)``.

    Returns
    -------
    float
        Approximate diagonal Hamiltonian element ``<j|H|j>``.
    """

    j = int(j)
    norb = int(drt.norb)

    h1e = np.asarray(h1e, dtype=np.float64)
    if h1e.shape != (norb, norb):
        raise ValueError("h1e has wrong shape")

    if eri_ppqq is None or eri_pqqp is None:
        eri4 = _restore_eri_4d(eri, norb).astype(np.float64, copy=False)
        if eri_ppqq is None:
            eri_ppqq = np.einsum("iijj->ij", eri4)
        if eri_pqqp is None:
            eri_pqqp = np.einsum("ijji->ij", eri4)
    else:
        eri_ppqq = np.asarray(eri_ppqq, dtype=np.float64)
        eri_pqqp = np.asarray(eri_pqqp, dtype=np.float64)
        if eri_ppqq.shape != (norb, norb):
            raise ValueError("eri_ppqq has wrong shape")
        if eri_pqqp.shape != (norb, norb):
            raise ValueError("eri_pqqp has wrong shape")

    steps = drt.index_to_path(j).astype(np.int8, copy=False)
    occ = _STEP_TO_OCC[steps].astype(np.int8, copy=False)
    neleca = (int(drt.nelec) + int(drt.twos_target)) // 2
    nelecb = int(drt.nelec) - neleca
    alpha, beta = _assign_spin_occupations(occ, neleca=neleca, nelecb=nelecb)

    n = (alpha + beta).astype(np.float64)
    alpha_f = alpha.astype(np.float64)
    beta_f = beta.astype(np.float64)

    hdiag = float(np.dot(np.diag(h1e), n))

    ecoul = 0.5 * float(n @ eri_ppqq @ n)
    exa = -0.5 * float(alpha_f @ eri_pqqp @ alpha_f)
    exb = -0.5 * float(beta_f @ eri_pqqp @ beta_f)
    return hdiag + ecoul + exa + exb


# ---------------------------------------------------------------------------
# Structure-only row oracle (Phase 0)
# ---------------------------------------------------------------------------


def connected_row_structure_only(
    drt: DRT,
    j: int,
    *,
    max_out: int = 200_000,
    max_occ_diff_orbitals: int = 4,
) -> np.ndarray:
    """Return candidate DRT indices connected to CSF j by 1-/2-body structure.

    Phase-0 oracle: identifies *potential* connections using only spatial
    occupancy differences. For a 2-body Hamiltonian, occupations can differ
    on at most 4 orbitals.

    Parameters
    ----------
    drt : DRT
        The Distinct Row Table.
    j : int
        CSF index.
    max_out : int, optional
        Maximum number of output indices. Default 200000.
    max_occ_diff_orbitals : int, optional
        Maximum number of orbitals that may differ in occupation.
        Default 4 (appropriate for 2-body Hamiltonians).

    Returns
    -------
    np.ndarray
        int32 array of candidate CSF indices (``j`` is always first).

    Raises
    ------
    IndexError
        If ``j`` is out of range.
    ValueError
        If the output exceeds ``max_out``.
    """

    j = int(j)
    ncsf = int(drt.ncsf)
    if j < 0 or j >= ncsf:
        raise IndexError(f"CSF index out of range: {j} (ncsf={ncsf})")
    max_out = int(max_out)
    if max_out <= 0:
        raise ValueError("max_out must be > 0")
    max_occ_diff_orbitals = int(max_occ_diff_orbitals)
    if max_occ_diff_orbitals < 0:
        raise ValueError("max_occ_diff_orbitals must be >= 0")

    occ_j = occ_table(drt)[j].astype(np.int8, copy=False)
    norb = int(drt.norb)
    groups = occ_groups(drt)

    # Generate reachable occupation patterns (0,1,2 per orbital) under 1- and 2-electron moves.
    # This is a conservative superset for 1-/2-body Hamiltonians; coupling-only changes are
    # covered by including the original occupation pattern.
    keys: set[bytes] = set()
    keys.add(bytes(occ_j.tolist()))

    occ_j_i = occ_j.astype(np.int16, copy=False)

    if max_occ_diff_orbitals >= 2:
        # 1-electron moves: pick src with occ>0 and dst with occ<2.
        src1 = np.nonzero(occ_j_i > 0)[0].tolist()
        dst1 = np.nonzero(occ_j_i < 2)[0].tolist()
        for src in src1:
            for dst in dst1:
                if dst == src:
                    continue
                occ_new = occ_j_i.copy()
                occ_new[src] -= 1
                occ_new[dst] += 1
                keys.add(bytes(occ_new.astype(np.int8).tolist()))

    if max_occ_diff_orbitals >= 4:
        # 2-electron moves: remove 2 electrons (2 from one orb, or 1+1 from two orbs),
        # then add 2 electrons with the post-removal occupations.
        src_pop = np.nonzero(occ_j_i > 0)[0].tolist()
        rm_opts: list[list[tuple[int, int]]] = []
        for a in src_pop:
            if occ_j_i[a] >= 2:
                rm_opts.append([(a, 2)])
        for ia, a in enumerate(src_pop):
            for b in src_pop[ia + 1 :]:
                rm_opts.append([(a, 1), (b, 1)])

        for rm in rm_opts:
            occ_rem = occ_j_i.copy()
            for p, dn in rm:
                occ_rem[p] -= dn

            add_pop1 = np.nonzero(occ_rem < 2)[0].tolist()
            add_pop2 = np.nonzero(occ_rem == 0)[0].tolist()

            # +2 into one orbital
            for c in add_pop2:
                occ_new = occ_rem.copy()
                occ_new[c] += 2
                keys.add(bytes(occ_new.astype(np.int8).tolist()))

            # +1 +1 into two orbitals
            for ic, c in enumerate(add_pop1):
                for d in add_pop1[ic + 1 :]:
                    occ_new = occ_rem.copy()
                    occ_new[c] += 1
                    occ_new[d] += 1
                    keys.add(bytes(occ_new.astype(np.int8).tolist()))

    idx_parts: list[np.ndarray] = []
    for key in keys:
        part = groups.get(key)
        if part is not None:
            idx_parts.append(part)
    if not idx_parts:
        idx = np.asarray([j], dtype=np.int32)
    else:
        idx = np.concatenate(idx_parts).astype(np.int32, copy=False)
        if int(idx[0]) != j:
            pos = int(np.nonzero(idx == j)[0][0])
            if pos != 0:
                idx = idx.copy()
                idx[[0, pos]] = idx[[pos, 0]]

    if idx.size > max_out:
        raise ValueError(f"oracle produced {idx.size} entries > max_out={max_out}")
    return idx


# ---------------------------------------------------------------------------
# Full connected row oracle
# ---------------------------------------------------------------------------


def connected_row(drt: DRT, h1e, eri, j: int, max_out: int = 200_000):
    """Return ``(i_idx, hij)`` for CSF j.

    Pure-Python prototype using 1-body GUGA segment tables and a naive
    2-body expansion in spin-free generators.

    Parameters
    ----------
    drt : DRT
        The Distinct Row Table.
    h1e : array_like
        One-electron integrals, shape ``(norb, norb)``.
    eri : array_like
        Two-electron integrals (packed or full form).
    j : int
        CSF index for which to compute the Hamiltonian row.
    max_out : int, optional
        Maximum number of output entries. Default 200000.

    Returns
    -------
    i_idx : np.ndarray
        int32 array of CSF indices (``j`` is always first).
    hij : np.ndarray
        float64 array of Hamiltonian matrix elements ``<i|H|j>``.

    Notes
    -----
    Implements the spin-free Hamiltonian:

    .. math::

        H = \\sum_{pq} h_{pq} E_{pq}
          + \\frac{1}{2} \\sum_{pqrs} (pq|rs)(E_{pq} E_{rs} - \\delta_{qr} E_{ps})

    The contraction term ``-delta_qr E_ps`` is folded into an effective
    one-body matrix ``h_eff``, and the ``r == s`` slice of the two-body
    product is absorbed as ``h_eff += 0.5 * sum_r (pq|rr) * occ_r(j)``.
    """

    j = int(j)
    max_out = int(max_out)
    if max_out < 1:
        raise ValueError("max_out must be >= 1")

    norb = int(drt.norb)
    h1e = np.asarray(h1e, dtype=np.float64)
    if h1e.shape != (norb, norb):
        raise ValueError("h1e has wrong shape")

    cache = _get_epq_action_cache(drt)
    steps = cache.steps[int(j)]

    eri4 = _restore_eri_4d(eri, norb).astype(np.float64, copy=False)

    occ_j = _STEP_TO_OCC[steps].astype(np.int8, copy=False)

    # Spin-free Hamiltonian in generator form:
    #   H = Σ_pq h_pq E_pq + 1/2 Σ_pqrs (pq|rs) (E_pq E_rs - δ_qr E_ps)
    # Fold the contraction term into an effective 1-body coefficient matrix.
    h_eff = h1e - 0.5 * np.einsum("pqqs->ps", eri4)

    # Pull out the r==s slice of the 2-body product: E_pq E_rr scales |j> by occ_r(j).
    h_eff = h_eff + 0.5 * np.einsum("pqrr,r->pq", eri4, occ_j)

    ncsf = int(drt.ncsf)
    acc = np.zeros(ncsf, dtype=np.float64)
    acc[int(j)] = float(np.dot(np.diag(h_eff), occ_j.astype(np.float64)))

    # 1-body off-diagonal (p!=q) contributions with effective h_eff.
    # Restrict to moves E_pq that can act on |j>: occ[q]>0 and occ[p]<2.
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
    # Original prototype expanded Σ_{pqrs} (pq|rs) E_pq E_rs by iterating (r,s) then (p,q),
    # which is correct but slow. Here we group by the intermediate state |k> reached by E_rs
    # and contract the ERI slice with all rs-contributions via a small matrix multiplication.
    occ_all = occ_table(drt)
    eri_mat = (0.5 * eri4.reshape(norb * norb, norb * norb)).astype(np.float64, copy=False)

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

    # For each intermediate k: build g_pq = Σ_{rs} (1/2)(pq|rs) <k|E_rs|j> and apply it as a
    # one-body operator Σ_{pq} g_pq E_pq |k>.
    for k, rs_terms in by_k.items():
        rs_ids = np.asarray([t[0] for t in rs_terms], dtype=np.int32)
        rs_coeff = np.asarray([t[1] for t in rs_terms], dtype=np.float64)
        g_flat = eri_mat[:, rs_ids] @ rs_coeff
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
