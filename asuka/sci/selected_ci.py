from __future__ import annotations

"""Selected / selective CI (SCI) in a GUGA/DRT CSF basis.

This module implements a compact *selected CI* driver that
works directly with the native :class:`asuka.cuguga.drt.DRT` CSF index space.

The core idea is the standard variational + selection loop:

1) Start from a small CSF subset (the *variational space*).
2) Build/diagonalize the Hamiltonian restricted to that subset.
3) Use sparse row-oracle scans to estimate first-order amplitudes (or EN-PT2
   contributions) of external CSFs.
4) Add the most important external CSFs, repeat until convergence.

The implementation is intentionally simple and easy to audit/modify.
It is aimed at integration, regression testing, and method development.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception as _e:  # pragma: no cover
    sp = None  # type: ignore[assignment]
    spla = None  # type: ignore[assignment]

from asuka.cuguga.drt import DRT
from asuka.cuguga.screening import RowScreening
from asuka.cuguga.state_cache import DRTStateCache, get_state_cache
from asuka.integrals.df_integrals import DFMOIntegrals
from asuka.cuguga.oracle import _STEP_TO_OCC, _restore_eri_4d
from asuka.cuguga.oracle.sparse import connected_row_sparse, connected_row_sparse_df
from asuka.solver import GUGAFCISolver


@dataclass
class SCIResult:
    """Results from a selected-CI run."""

    e_var: np.ndarray  # (nroots,)
    e_pt2: np.ndarray  # (nroots,)
    e_tot: np.ndarray  # (nroots,) = e_var + e_pt2
    sel_idx: np.ndarray  # (nsel,) global CSF indices
    ci_sel: np.ndarray  # (nsel, nroots)
    ci_full: list[np.ndarray]  # length nroots; each is (ncsf,)
    history: list[dict[str, Any]]


def _connected_row(
    drt: DRT,
    h1e: np.ndarray,
    eri: Any,
    j: int,
    *,
    max_out: int,
    screening: RowScreening | None,
    state_cache: DRTStateCache | None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(eri, DFMOIntegrals):
        return connected_row_sparse_df(
            drt,
            h1e,
            eri,
            int(j),
            max_out=int(max_out),
            screening=screening,
            state_cache=state_cache,
        )
    return connected_row_sparse(
        drt,
        h1e,
        eri,
        int(j),
        max_out=int(max_out),
        screening=screening,
        state_cache=state_cache,
    )


def _make_hdiag_guess(drt: DRT, h1e: np.ndarray, eri: Any, *, state_cache: DRTStateCache) -> np.ndarray:
    """Vectorized determinant-based diagonal guess for all CSFs in a DRT.

    This mirrors :meth:`asuka.solver.GUGAFCISolver.make_hdiag`, but takes a
    pre-built DRT and state cache.
    """

    norb = int(drt.norb)
    ncsf = int(drt.ncsf)
    if ncsf <= 0:
        return np.zeros((0,), dtype=np.float64)

    h1e = np.asarray(h1e, dtype=np.float64)
    if h1e.shape != (norb, norb):
        raise ValueError("h1e has wrong shape")
    h1e_diag = np.diag(h1e)

    if isinstance(eri, DFMOIntegrals):
        l_full = np.asarray(eri.l_full, dtype=np.float64, order="C")
        pair_norm = np.asarray(eri.pair_norm, dtype=np.float64, order="C")

        diag_ids = (np.arange(norb, dtype=np.int32) * (norb + 1)).astype(np.int32, copy=False)
        l_diag = np.asarray(l_full[diag_ids], dtype=np.float64, order="C")
        eri_ppqq = l_diag @ l_diag.T  # (p p| q q)
        eri_pqqp = np.square(pair_norm.reshape(norb, norb))
    else:
        eri4 = _restore_eri_4d(eri, norb).astype(np.float64, copy=False)
        eri_ppqq = np.einsum("iijj->ij", eri4)
        eri_pqqp = np.einsum("ijji->ij", eri4)

    steps = np.asarray(state_cache.steps, dtype=np.int8)
    if steps.shape != (ncsf, norb):
        raise RuntimeError("internal error: invalid cached steps table shape")
    occ = _STEP_TO_OCC[steps]

    doubly = occ == 2
    singles = occ == 1

    neleca_det = (int(drt.nelec) + int(drt.twos_target)) // 2
    nelecb_det = int(drt.nelec) - neleca_det

    ndoubly = np.sum(doubly, axis=1, dtype=np.int32)
    alpha_need = np.asarray(neleca_det, dtype=np.int32) - ndoubly

    single_prefix = np.cumsum(singles, axis=1, dtype=np.int32)
    alpha_single = singles & (single_prefix <= alpha_need[:, None])
    beta_single = singles & (~alpha_single)

    alpha = (doubly | alpha_single).astype(np.float64)
    beta = (doubly | beta_single).astype(np.float64)
    n = alpha + beta

    hdiag = n @ h1e_diag
    tmp = n @ eri_ppqq
    hdiag += 0.5 * np.sum(tmp * n, axis=1)
    tmp_a = alpha @ eri_pqqp
    hdiag += -0.5 * np.sum(tmp_a * alpha, axis=1)
    tmp_b = beta @ eri_pqqp
    hdiag += -0.5 * np.sum(tmp_b * beta, axis=1)
    return np.asarray(hdiag, dtype=np.float64)


def _normalize_ci0(ci0: Any, *, nroots: int, ncsf: int) -> list[np.ndarray] | None:
    if ci0 is None:
        return None

    if isinstance(ci0, (list, tuple)):
        out: list[np.ndarray] = []
        for x in ci0[:nroots]:
            v = np.asarray(x, dtype=np.float64).ravel()
            if int(v.size) != int(ncsf):
                raise ValueError("ci0 has wrong length")
            out.append(np.ascontiguousarray(v))
        if not out:
            return None
        if len(out) < nroots:
            # Pad with zeros.
            for _ in range(nroots - len(out)):
                out.append(np.zeros(ncsf, dtype=np.float64))
        return out

    v = np.asarray(ci0, dtype=np.float64)
    if v.ndim == 1:
        if int(v.size) != int(ncsf):
            raise ValueError("ci0 has wrong length")
        out = [np.ascontiguousarray(v.ravel())]
        for _ in range(nroots - 1):
            out.append(np.zeros(ncsf, dtype=np.float64))
        return out

    if v.ndim == 2:
        if int(v.shape[1]) != int(ncsf):
            raise ValueError("ci0 has wrong shape")
        out = [np.ascontiguousarray(v[i].ravel()) for i in range(min(nroots, int(v.shape[0])))]
        for _ in range(nroots - len(out)):
            out.append(np.zeros(ncsf, dtype=np.float64))
        return out

    raise ValueError("unsupported ci0 format")


def _initial_selection(
    *,
    ncsf: int,
    nroots: int,
    init_ncsf: int,
    hdiag: np.ndarray,
    ci0_list: list[np.ndarray] | None,
) -> list[int]:
    init_ncsf = int(init_ncsf)
    if init_ncsf <= 0:
        init_ncsf = max(1, int(nroots))
    init_ncsf = min(init_ncsf, int(ncsf))

    sel: list[int] = []
    seen: set[int] = set()

    # 1) Seed from user-provided ci0 (largest coefficients).
    if ci0_list is not None:
        # Take a generous slice so we can form a union across roots.
        per_root = max(1, init_ncsf // max(1, nroots))
        per_root = max(per_root, min(64, init_ncsf))
        for r in range(nroots):
            v = np.asarray(ci0_list[r], dtype=np.float64)
            if not np.any(v):
                continue
            mag = np.abs(v)
            if int(mag.size) != int(ncsf):
                raise ValueError("internal error: ci0 length mismatch")
            if per_root >= ncsf:
                idx = np.argsort(mag)[::-1]
            else:
                idx = np.argpartition(mag, -per_root)[-per_root:]
                idx = idx[np.argsort(mag[idx])[::-1]]
            for i in idx.tolist():
                ii = int(i)
                if ii in seen:
                    continue
                sel.append(ii)
                seen.add(ii)
                if len(sel) >= init_ncsf:
                    break
            if len(sel) >= init_ncsf:
                break

    # 2) Fill remaining slots from low diagonal guesses.
    if len(sel) < init_ncsf:
        need = init_ncsf - len(sel)
        if need >= ncsf:
            idx = np.argsort(hdiag)
        else:
            idx = np.argpartition(hdiag, need)[:need]
            idx = idx[np.argsort(hdiag[idx])]
        for i in idx.tolist():
            ii = int(i)
            if ii in seen:
                continue
            sel.append(ii)
            seen.add(ii)
            if len(sel) >= init_ncsf:
                break

    # Ensure at least nroots states are present.
    if len(sel) < nroots:
        for i in range(int(nroots)):
            if i not in seen and i < ncsf:
                sel.append(int(i))
                seen.add(int(i))
            if len(sel) >= nroots:
                break

    return sel


def _build_variational_hamiltonian(
    drt: DRT,
    h1e: np.ndarray,
    eri: Any,
    *,
    sel: Sequence[int],
    loc: np.ndarray,
    max_out: int,
    screening: RowScreening | None,
    state_cache: DRTStateCache | None,
) -> "sp.csr_matrix":
    if sp is None:  # pragma: no cover
        raise RuntimeError("scipy is required for selected_ci")

    nsel = int(len(sel))
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for col, j in enumerate(sel):
        i_idx, hij = _connected_row(
            drt,
            h1e,
            eri,
            int(j),
            max_out=max_out,
            screening=screening,
            state_cache=state_cache,
        )
        for i, v in zip(i_idx.tolist(), hij.tolist()):
            row = int(loc[int(i)])
            if row < 0:
                continue
            rows.append(row)
            cols.append(int(col))
            data.append(float(v))

    h = sp.coo_matrix((np.asarray(data, dtype=np.float64), (rows, cols)), shape=(nsel, nsel)).tocsr()
    # Numerical symmetrization (oracle paths should already be symmetric up to FP).
    h = (h + h.T) * 0.5
    h.eliminate_zeros()
    return h


def _solve_subspace(
    h: "sp.csr_matrix",
    *,
    nroots: int,
    dense_limit: int,
    eigsh_tol: float,
    v0: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if sp is None or spla is None:  # pragma: no cover
        raise RuntimeError("scipy is required for selected_ci")

    nsel = int(h.shape[0])
    nroots = int(nroots)
    if nsel <= 0:
        raise ValueError("empty variational space")
    if nroots < 1:
        raise ValueError("nroots must be >= 1")
    if nroots > nsel:
        raise ValueError("nroots must be <= len(sel)")

    dense_limit = int(dense_limit)
    if dense_limit <= 0:
        dense_limit = 1024

    use_dense = nsel <= dense_limit or nsel <= max(4 * nroots + 16, nroots + 2)
    if use_dense:
        hd = np.asarray(h.toarray(), dtype=np.float64)
        evals, evecs = np.linalg.eigh(hd)
        evals = np.asarray(evals[:nroots], dtype=np.float64)
        evecs = np.asarray(evecs[:, :nroots], dtype=np.float64)
        return evals, evecs

    # ARPACK (eigsh) for larger spaces.
    # Note: ARPACK's convergence can be sensitive; keep parameters conservative.
    evals, evecs = spla.eigsh(
        h,
        k=int(nroots),
        which="SA",
        tol=float(eigsh_tol),
        v0=None if v0 is None else np.asarray(v0, dtype=np.float64).ravel(),
    )
    order = np.argsort(evals)
    evals = np.asarray(evals[order], dtype=np.float64)
    evecs = np.asarray(evecs[:, order], dtype=np.float64)
    return evals, evecs


def _select_external(
    drt: DRT,
    h1e: np.ndarray,
    eri: Any,
    *,
    sel: Sequence[int],
    loc: np.ndarray,
    c_sel: np.ndarray,
    e_var: np.ndarray,
    hdiag: np.ndarray,
    max_add: int,
    select_threshold: float | None,
    denom_floor: float,
    max_out: int,
    screening: RowScreening | None,
    state_cache: DRTStateCache | None,
    select_screen_contrib: float,
) -> tuple[list[int], np.ndarray]:
    """Return (new_indices, e_pt2) for the current variational solution."""

    nroots = int(e_var.size)
    denom_floor = float(denom_floor)
    if denom_floor < 0.0:
        raise ValueError("denom_floor must be >= 0")

    # Accumulate external couplings: p_i = sum_j H_ij c_j.
    ext: dict[int, np.ndarray] = {}

    for col, j in enumerate(sel):
        cj = np.asarray(c_sel[col, :], dtype=np.float64)
        max_cj = float(np.max(np.abs(cj)))
        if max_cj == 0.0:
            continue

        i_idx, hij = _connected_row(
            drt,
            h1e,
            eri,
            int(j),
            max_out=max_out,
            screening=screening,
            state_cache=state_cache,
        )

        for i, v in zip(i_idx.tolist(), hij.tolist()):
            ii = int(i)
            if int(loc[ii]) >= 0:
                continue
            vv = float(v)
            if select_screen_contrib > 0.0 and abs(vv) * max_cj < select_screen_contrib:
                continue
            acc = ext.get(ii)
            if acc is None:
                ext[ii] = vv * cj
            else:
                acc += vv * cj

    if not ext:
        return [], np.zeros((nroots,), dtype=np.float64)

    # Score candidates.
    cand_i: list[int] = []
    cand_w: list[float] = []
    e_pt2 = np.zeros((nroots,), dtype=np.float64)

    for ii, p in ext.items():
        denom = np.asarray(e_var - float(hdiag[int(ii)]), dtype=np.float64)
        if denom_floor > 0.0:
            small = np.abs(denom) < denom_floor
            if np.any(small):
                denom = denom.copy()
                # Preserve sign when possible; if denom==0 set positive floor.
                denom[small] = np.where(denom[small] >= 0.0, denom_floor, -denom_floor)

        c1 = p / denom
        w = float(np.max(np.abs(c1)))
        if w == 0.0 or not np.isfinite(w):
            continue
        cand_i.append(int(ii))
        cand_w.append(w)
        # Epstein-Nesbet PT2 correction estimate per root.
        e_pt2 += (p * p) / denom

    if not cand_i:
        return [], e_pt2

    w_arr = np.asarray(cand_w, dtype=np.float64)
    i_arr = np.asarray(cand_i, dtype=np.int64)

    # Determine which candidates to add.
    max_add = int(max_add)
    if max_add <= 0:
        return [], e_pt2
    max_add = min(max_add, int(i_arr.size))

    if select_threshold is not None:
        thr = float(select_threshold)
        if thr < 0.0:
            raise ValueError("select_threshold must be >= 0")
        keep = np.nonzero(w_arr >= thr)[0]
        if int(keep.size) == 0:
            return [], e_pt2
        if int(keep.size) > max_add:
            # Take the top max_add among those exceeding threshold.
            k = max_add
            sub = keep[np.argpartition(w_arr[keep], -k)[-k:]]
            keep = sub
    else:
        # Take the top max_add by weight.
        keep = np.argpartition(w_arr, -max_add)[-max_add:]

    # Order selected candidates by descending weight for determinism.
    keep = keep[np.argsort(w_arr[keep])[::-1]]
    new_idx = [int(x) for x in i_arr[keep].tolist()]
    return new_idx, e_pt2


def selected_ci(
    drt: DRT,
    h1e: np.ndarray,
    eri: Any,
    *,
    hdiag: np.ndarray | None = None,
    ci0: Any = None,
    ecore: float = 0.0,
    nroots: int = 1,
    init_ncsf: int = 256,
    max_ncsf: int = 50_000,
    add_ncsf: int = 2_000,
    max_iter: int = 20,
    select_threshold: float | None = None,
    denom_floor: float = 1e-12,
    max_out: int = 200_000,
    screening: RowScreening | None = None,
    use_state_cache: bool = True,
    dense_limit: int = 4096,
    eigsh_tol: float = 1e-12,
    select_screen_contrib: float = 0.0,
    verbose: int = 0,
) -> SCIResult:
    """Run a basic selected-CI (SCI) calculation in a CSF basis.

    Parameters
    ----------
    drt
        The DRT defining the full CSF space.
    h1e, eri
        Active-space one- and two-electron integrals.
    hdiag
        Optional determinant-based diagonal guess (length ncsf) used for
        denominators in the selection step. If not provided, it is computed
        using a vectorized guess based on cached DRT steps.
    ci0
        Optional initial CI vector(s) in the *full* CSF space.
    ecore
        Constant energy shift (added to reported energies).
    nroots
        Number of roots to compute/select simultaneously.
    init_ncsf, max_ncsf, add_ncsf
        Controls the variational space size and growth.
    max_iter
        Maximum selection iterations.
    select_threshold
        If not None, add all external CSFs with estimated first-order amplitude
        >= threshold (capped by ``add_ncsf`` per iteration). If None, add the
        top ``add_ncsf`` CSFs by estimated amplitude.
    denom_floor
        Lower bound on |E - H_ii| in the selection denominator.
    screening
        Optional :class:`~asuka.cuguga.screening.RowScreening` passed to the row
        oracle. This can be used as an approximate/accelerated mode.
    select_screen_contrib
        Optional extra screening in the selection accumulation step: skip an
        external contribution if ``|H_ij| * max_r |c_j(r)| < select_screen_contrib``.
    """

    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be >= 1")

    ncsf = int(drt.ncsf)
    if ncsf < 1:
        raise ValueError("drt.ncsf must be >= 1")
    if nroots > ncsf:
        raise ValueError("nroots must be <= drt.ncsf")

    max_ncsf = int(max_ncsf)
    if max_ncsf < nroots:
        raise ValueError("max_ncsf must be >= nroots")
    max_ncsf = min(max_ncsf, ncsf)

    # Cache CSF step/node tables once (strongly recommended for repeated oracle calls).
    state_cache = get_state_cache(drt) if bool(use_state_cache) else None

    if hdiag is None:
        if state_cache is None:
            raise ValueError("hdiag=None requires use_state_cache=True (or provide hdiag explicitly)")
        hdiag = _make_hdiag_guess(drt, h1e, eri, state_cache=state_cache)
    else:
        hdiag = np.asarray(hdiag, dtype=np.float64).ravel()
        if int(hdiag.size) != ncsf:
            raise ValueError("hdiag has wrong length")

    ci0_list = _normalize_ci0(ci0, nroots=nroots, ncsf=ncsf)

    sel_list = _initial_selection(
        ncsf=ncsf,
        nroots=nroots,
        init_ncsf=int(init_ncsf),
        hdiag=hdiag,
        ci0_list=ci0_list,
    )

    # Basis mapping: loc[global_idx] = local_idx in `sel`, or -1.
    loc = -np.ones(ncsf, dtype=np.int32)
    sel: list[int] = []
    for i in sel_list:
        ii = int(i)
        if ii < 0 or ii >= ncsf:
            continue
        if int(loc[ii]) >= 0:
            continue
        loc[ii] = int(len(sel))
        sel.append(ii)
        if len(sel) >= max_ncsf:
            break

    history: list[dict[str, Any]] = []

    # Start vector for eigsh (ground state only); carry between iterations.
    v0: np.ndarray | None = None
    prev_e_var: np.ndarray | None = None

    for it in range(1, int(max_iter) + 1):
        nsel = int(len(sel))
        if nsel < nroots:
            raise RuntimeError("internal error: variational space smaller than nroots")

        if verbose:
            print(f"[SCI] iter {it}: nsel={nsel}")

        # Assemble and diagonalize the variational Hamiltonian.
        h_var = _build_variational_hamiltonian(
            drt,
            h1e,
            eri,
            sel=sel,
            loc=loc,
            max_out=max_out,
            screening=screening,
            state_cache=state_cache,
        )

        # Use the previous ground-state vector as an initial guess for ARPACK when possible.
        if v0 is not None and int(v0.size) != nsel:
            # sel only grows; pad old v0 with zeros.
            if int(v0.size) < nsel:
                v0 = np.pad(v0, (0, nsel - int(v0.size)))
            else:
                v0 = None

        e_var, c_sel = _solve_subspace(
            h_var,
            nroots=nroots,
            dense_limit=dense_limit,
            eigsh_tol=eigsh_tol,
            v0=v0,
        )

        # Store a v0 guess for the next iteration.
        v0 = np.asarray(c_sel[:, 0], dtype=np.float64).copy()

        # Selection step.
        nleft = max_ncsf - nsel
        max_add = min(int(add_ncsf), int(nleft))

        new_idx, e_pt2 = _select_external(
            drt,
            h1e,
            eri,
            sel=sel,
            loc=loc,
            c_sel=c_sel,
            e_var=e_var,
            hdiag=hdiag,
            max_add=max_add,
            select_threshold=select_threshold,
            denom_floor=denom_floor,
            max_out=max_out,
            screening=screening,
            state_cache=state_cache,
            select_screen_contrib=float(select_screen_contrib),
        )

        e_tot = np.asarray(e_var + e_pt2, dtype=np.float64)

        rec: dict[str, Any] = {
            "iter": int(it),
            "nsel": int(nsel),
            "nadd": int(len(new_idx)),
            "e_var": np.asarray(e_var + float(ecore), dtype=np.float64).copy(),
            "e_pt2": np.asarray(e_pt2, dtype=np.float64).copy(),
            "e_tot": np.asarray(e_tot + float(ecore), dtype=np.float64).copy(),
        }
        if prev_e_var is None:
            rec["de_var"] = None
        else:
            rec["de_var"] = np.asarray(e_var - prev_e_var, dtype=np.float64)
        history.append(rec)

        prev_e_var = e_var.copy()

        if not new_idx or nleft <= 0:
            break

        # Add new CSFs (append order).
        for ii in new_idx:
            if int(loc[int(ii)]) >= 0:
                continue
            loc[int(ii)] = int(len(sel))
            sel.append(int(ii))
            if len(sel) >= max_ncsf:
                break

        if len(sel) >= max_ncsf:
            break

    # Final solve on the final selected space.
    nsel = int(len(sel))
    h_var = _build_variational_hamiltonian(
        drt,
        h1e,
        eri,
        sel=sel,
        loc=loc,
        max_out=max_out,
        screening=screening,
        state_cache=state_cache,
    )
    e_var, c_sel = _solve_subspace(
        h_var,
        nroots=nroots,
        dense_limit=dense_limit,
        eigsh_tol=eigsh_tol,
        v0=None,
    )
    _new_idx, e_pt2 = _select_external(
        drt,
        h1e,
        eri,
        sel=sel,
        loc=loc,
        c_sel=c_sel,
        e_var=e_var,
        hdiag=hdiag,
        max_add=0,
        select_threshold=None,
        denom_floor=denom_floor,
        max_out=max_out,
        screening=screening,
        state_cache=state_cache,
        select_screen_contrib=float(select_screen_contrib),
    )
    e_tot = np.asarray(e_var + e_pt2, dtype=np.float64)

    # Expand CI vectors to full size (dense) for compatibility with downstream code.
    sel_idx = np.asarray(sel, dtype=np.int32)
    ci_full: list[np.ndarray] = []
    for r in range(nroots):
        v = np.zeros(ncsf, dtype=np.float64)
        v[sel_idx] = np.asarray(c_sel[:, r], dtype=np.float64)
        ci_full.append(np.ascontiguousarray(v))

    return SCIResult(
        e_var=np.asarray(e_var + float(ecore), dtype=np.float64),
        e_pt2=np.asarray(e_pt2, dtype=np.float64),
        e_tot=np.asarray(e_tot + float(ecore), dtype=np.float64),
        sel_idx=sel_idx,
        ci_sel=np.asarray(c_sel, dtype=np.float64),
        ci_full=ci_full,
        history=history,
    )


class GUGASelectedCISolver(GUGAFCISolver):
    """A drop-in-ish FCI solver implementing basic selected CI in a CSF basis.

    This class follows the PySCF FCI solver calling convention (``kernel``
    signature) but computes an approximate CI solution by selecting a subset of
    CSFs iteratively.

    Notes
    -----
    * The returned CI vector is a **dense** vector in the full CSF space
      (unselected components are zero). This keeps compatibility with existing
      RDM builders.
    * The attribute :attr:`last_sci_result` stores the full run record,
      including the PT2 estimate.
    """

    # Selection controls (may be overridden per-call via keyword arguments).
    sci_init_ncsf: int = 256
    sci_max_ncsf: int = 50_000
    sci_add_ncsf: int = 2_000
    sci_max_iter: int = 20
    sci_select_threshold: float | None = None
    sci_denom_floor: float = 1e-12
    sci_dense_limit: int = 4096
    sci_eigsh_tol: float = 1e-12
    sci_select_screen_contrib: float = 0.0
    sci_use_state_cache: bool = True

    # Row-oracle controls.
    sci_max_out: int = 200_000
    sci_row_screening: RowScreening | None = None

    last_sci_result: SCIResult | None = None

    def kernel(
        self,
        h1e,
        eri,
        norb: int,
        nelec: int | tuple[int, int],
        ci0=None,
        ecore: float = 0.0,
        nroots: int = 1,
        **kwargs,
    ):
        norb = int(norb)
        neleca, nelecb, nelec_total, _sz_twos = self._normalize_nelec(nelec)
        twos = self._get_twos_target(neleca, nelecb)

        orbsym = kwargs.get("orbsym", getattr(self, "orbsym", None))
        wfnsym = kwargs.get("wfnsym", getattr(self, "wfnsym", None))
        drt = self._get_drt(norb, nelec_total, twos, orbsym=orbsym, wfnsym=wfnsym)

        # Build the diagonal guess once.
        # (This also builds the state cache, which helps repeated row-oracle scans.)
        hdiag = self.make_hdiag(h1e, eri, norb, nelec, orbsym=orbsym, wfnsym=wfnsym)

        verbose_val = kwargs.get("verbose", getattr(self, "verbose", 0))
        if hasattr(verbose_val, "verbose"):  # PySCF Logger
            verbose_val = verbose_val.verbose
        
        res = selected_ci(
            drt,
            h1e,
            eri,
            hdiag=hdiag,
            ci0=ci0,
            ecore=float(ecore),
            nroots=int(nroots),
            init_ncsf=int(kwargs.get("sci_init_ncsf", getattr(self, "sci_init_ncsf", 256))),
            max_ncsf=int(kwargs.get("sci_max_ncsf", getattr(self, "sci_max_ncsf", 50_000))),
            add_ncsf=int(kwargs.get("sci_add_ncsf", getattr(self, "sci_add_ncsf", 2_000))),
            max_iter=int(kwargs.get("sci_max_iter", getattr(self, "sci_max_iter", 20))),
            select_threshold=kwargs.get("sci_select_threshold", getattr(self, "sci_select_threshold", None)),
            denom_floor=float(kwargs.get("sci_denom_floor", getattr(self, "sci_denom_floor", 1e-12))),
            max_out=int(kwargs.get("sci_max_out", getattr(self, "sci_max_out", 200_000))),
            screening=kwargs.get("sci_row_screening", getattr(self, "sci_row_screening", None)),
            use_state_cache=bool(kwargs.get("sci_use_state_cache", getattr(self, "sci_use_state_cache", True))),
            dense_limit=int(kwargs.get("sci_dense_limit", getattr(self, "sci_dense_limit", 4096))),
            eigsh_tol=float(kwargs.get("sci_eigsh_tol", getattr(self, "sci_eigsh_tol", 1e-12))),
            select_screen_contrib=float(
                kwargs.get("sci_select_screen_contrib", getattr(self, "sci_select_screen_contrib", 0.0))
            ),
            verbose=int(verbose_val or 0),
        )
        self.last_sci_result = res

        # Return the variational energy by default (PT2 is stored in `last_sci_result`).
        if int(nroots) == 1:
            return float(res.e_var[0]), res.ci_full[0]
        return np.asarray(res.e_var, dtype=np.float64), res.ci_full
