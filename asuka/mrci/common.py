from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

from asuka.cuguga.drt import build_drt
from asuka.solver import GUGAFCISolver


def embed_ci_with_docc_prefix(
    *,
    ci_act: np.ndarray,
    n_docc: int,
    n_act: int,
    nelec_act: int,
    twos: int,
    orbsym_act: Sequence[int] | None,
    orbsym_full: Sequence[int] | None,
    wfnsym: int | None,
) -> np.ndarray:
    """Embed an active-space CI vector into a larger internal space with a fixed DOCC prefix."""

    n_docc_i = int(n_docc)
    if n_docc_i <= 0:
        return np.asarray(ci_act, dtype=np.float64)

    n_act_i = int(n_act)
    nelec_act_i = int(nelec_act)
    twos_i = int(twos)
    norb_small = n_act_i
    norb_large = n_docc_i + n_act_i
    nelec_large = nelec_act_i + 2 * n_docc_i

    drt_small = build_drt(
        norb=norb_small,
        nelec=nelec_act_i,
        twos_target=twos_i,
        orbsym=orbsym_act,
        wfnsym=wfnsym,
    )
    drt_large = build_drt(
        norb=norb_large,
        nelec=nelec_large,
        twos_target=twos_i,
        orbsym=orbsym_full,
        wfnsym=wfnsym,
    )

    ci_act_f64 = np.asarray(ci_act, dtype=np.float64).ravel()
    if int(ci_act_f64.size) != int(drt_small.ncsf):
        raise ValueError(f"ci_act has wrong size {int(ci_act_f64.size)} (expected {int(drt_small.ncsf)})")

    out = np.zeros(int(drt_large.ncsf), dtype=np.float64)
    steps_full = np.empty(norb_large, dtype=np.int8)
    steps_full[:n_docc_i] = 3  # D steps (doubly occupied)
    for j in range(int(drt_small.ncsf)):
        steps_act = drt_small.index_to_path(int(j))
        steps_full[n_docc_i:] = steps_act
        J = int(drt_large.path_to_index(steps_full))
        out[J] = float(ci_act_f64[j])
    return out


def compute_cas_reference_energy_df(
    *,
    h1e_corr: np.ndarray,
    l_full: Any,
    ecore: float,
    ci_cas: np.ndarray,
    n_act: int,
    nelec: int,
    twos: int,
    orbsym_act: Sequence[int] | None = None,
    wfnsym: int | None = None,
) -> float:
    """Compute E_ref (total) using DF/Cholesky vectors `l_full`.

    Notes
    -----
    `l_full` is expected to store ordered-pair vectors in shape (norb*norb, naux),
    consistent with :class:`asuka.integrals.df_integrals.DFMOIntegrals` and
    :class:`asuka.integrals.df_integrals.DeviceDFMOIntegrals`.
    """

    n_act = int(n_act)
    if n_act < 0:
        raise ValueError("n_act must be >= 0")

    h1e_corr = np.asarray(h1e_corr, dtype=np.float64)
    norb = int(h1e_corr.shape[0])
    if h1e_corr.shape != (norb, norb):
        raise ValueError("h1e_corr must be square")

    cas = GUGAFCISolver(twos=int(twos), orbsym=orbsym_act, wfnsym=wfnsym)
    gamma, dm2 = cas.make_rdm12(ci_cas, norb=n_act, nelec=int(nelec))
    gamma = np.asarray(gamma, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)

    h_int = np.asarray(h1e_corr[:n_act, :n_act], dtype=np.float64)
    e1 = float(np.einsum("pq,pq->", h_int, gamma, optimize=True))

    if n_act == 0:
        return float(ecore + e1)

    # Build ordered-pair IDs for the internal block (p,q in [0,n_act)).
    p = np.arange(n_act, dtype=np.int64)
    pq_ids = (p[:, None] * int(norb) + p[None, :]).reshape(n_act * n_act)

    dm2_mat = dm2.reshape(n_act * n_act, n_act * n_act)

    try:
        import cupy as cp  # noqa: PLC0415

        if isinstance(l_full, cp.ndarray):
            l_int = cp.take(cp.asarray(l_full, dtype=cp.float64), cp.asarray(pq_ids, dtype=cp.int64), axis=0)
            d = cp.asarray(dm2_mat, dtype=cp.float64)
            tmp = d @ l_int
            e2 = 0.5 * cp.sum(l_int * tmp)
            return float(ecore + e1 + float(cp.asnumpy(e2)))
    except Exception:
        pass

    l_full_np = np.asarray(l_full, dtype=np.float64, order="C")
    if l_full_np.ndim != 2 or int(l_full_np.shape[0]) != int(norb) * int(norb):
        raise ValueError("l_full must have shape (norb*norb, naux)")
    l_int_np = np.asarray(l_full_np[pq_ids], dtype=np.float64, order="C")
    tmp_np = dm2_mat @ l_int_np
    e2 = 0.5 * float(np.einsum("pL,pL->", l_int_np, tmp_np, optimize=True))
    return float(ecore + e1 + e2)


def assign_roots_by_overlap(
    overlap: np.ndarray,
    *,
    method: Literal["hungarian", "greedy"] = "hungarian",
) -> np.ndarray:
    """Assign MRCI roots to reference states by maximum overlap.

    Parameters
    ----------
    overlap
        Overlap matrix with shape (nref, nroots), where ``overlap[k, i] = |<ref_k|root_i>|^2``.
    method
        Assignment algorithm. ``"hungarian"`` solves a linear-sum assignment problem (unique mapping),
        while ``"greedy"`` picks the largest overlap iteratively.

    Returns
    -------
    np.ndarray
        Integer array of length nref where ``roots[k]`` is the assigned root index for
        reference state k. Indices are unique.
    """

    overlap = np.asarray(overlap, dtype=np.float64)
    if overlap.ndim != 2:
        raise ValueError("overlap must be a 2D array")
    nref, nroots = map(int, overlap.shape)
    if nref == 0:
        return np.zeros((0,), dtype=np.int64)
    if nroots < nref:
        raise ValueError(f"need nroots >= nref for assignment (got nroots={nroots}, nref={nref})")

    method_s = str(method).strip().lower()
    if method_s not in ("hungarian", "greedy"):
        raise ValueError("method must be 'hungarian' or 'greedy'")

    if method_s == "greedy":
        roots = -np.ones(nref, dtype=np.int64)
        used = np.zeros(nroots, dtype=bool)
        row_order = np.argsort(np.max(overlap, axis=1))[::-1]
        for k in row_order.tolist():
            cols = np.argsort(overlap[k])[::-1]
            for i in cols.tolist():
                if not bool(used[i]):
                    roots[k] = int(i)
                    used[i] = True
                    break
            if int(roots[k]) < 0:
                raise RuntimeError("failed to assign unique roots (greedy)")
        return roots

    try:
        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

        row, col = linear_sum_assignment(-overlap)
        if int(row.size) != nref:
            raise RuntimeError("unexpected assignment output size")
        order = np.argsort(row)
        roots = np.asarray(col[order], dtype=np.int64)
        if roots.shape != (nref,):
            raise RuntimeError("unexpected assignment output shape")
        if len(set(int(i) for i in roots.tolist())) != nref:
            raise RuntimeError("assignment produced duplicate roots")
        return roots
    except Exception:
        roots = -np.ones(nref, dtype=np.int64)
        used = np.zeros(nroots, dtype=bool)
        row_order = np.argsort(np.max(overlap, axis=1))[::-1]
        for k in row_order.tolist():
            cols = np.argsort(overlap[k])[::-1]
            for i in cols.tolist():
                if not bool(used[i]):
                    roots[k] = int(i)
                    used[i] = True
                    break
            if int(roots[k]) < 0:
                raise RuntimeError("failed to assign unique roots (fallback)")
        return roots
