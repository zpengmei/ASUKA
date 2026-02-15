from __future__ import annotations

"""Paper-faithful (RSI-style) multi-root FCI-FRI for excited states.

This module implements the randomized subspace iteration (RSI) variant used
by the FCI-FRI excited-state algorithm:

    X^{(t+1)} = (I - eps H) X^{(t)} [G^{(t)}]^{-1}

where the (small) stabilization matrix schedule G^{(t)} is designed to avoid
direct dot products between *random* iterate vectors.

This implementation is written to work with cuGUGA's GUGA/DRT Hamiltonian
action (spawn/oracle). It is intended as a "reference-oriented" path.

Notes
-----
- CPU path: fully implemented.
- CUDA path: projector + RSI evaluation (`S=U^T X`, stochastic `K=U^T H X`)
  are executed through the CUDA workspace backend. The Appendix-D
  stabilization/right-mixing remains the same sparse-column host logic.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Tuple
import warnings

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.state_cache import DRTStateCache, get_state_cache
from asuka.cuguga.oracle.sparse import connected_row_sparse
from .compress import compress_phi_pivot_resample, compress_phi_pivotal
from .omp import maybe_set_openmp_threads
from .projector import initiator_threshold, projector_step
from .sparse import coalesce_coo_i32_f64
from .subspace import axpy_sparse, dot_sparse

SparseCol = Tuple[np.ndarray, np.ndarray]  # (idx:int32[nnz], val:float64[nnz]) sorted unique


@dataclass(frozen=True)
class FCIFRIRSIResult:
    """Results from `run_fcifri_rsi`."""

    idx: List[np.ndarray]
    val: List[np.ndarray]
    iters: np.ndarray
    energies_inst: np.ndarray
    energies_avg: np.ndarray
    s_avg: np.ndarray
    k_avg: np.ndarray
    nsample: int
    backend: str


RSIEvalHook = Callable[
    [
        int,  # it
        np.ndarray,  # eval eigenvalues (sorted)
        np.ndarray,  # eval eigenvectors (columns correspond to eigenvalues)
        List[SparseCol],  # X_cols snapshot (iterate basis)
        List[SparseCol] | None,  # ritz_cols = normalize(X @ eval_eigvecs) (per-root), optional
        str,  # basis: "inst" or "avg"
        int,  # nsample used in averages
    ],
    None,
]


def _permute_vecs_cols(v: np.ndarray, perm: Sequence[int]) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    perm_a = np.asarray(list(perm), dtype=np.int64).ravel()
    if v.ndim != 2:
        raise ValueError("v must be 2D")
    if int(perm_a.size) != int(v.shape[1]):
        raise ValueError("perm size mismatch for v columns")
    return np.asarray(v[:, perm_a], dtype=np.float64, order="C")


def _permute_cols(cols: Sequence[SparseCol], perm: Sequence[int]) -> List[SparseCol]:
    perm_a = [int(x) for x in list(perm)]
    if int(len(perm_a)) != int(len(cols)):
        raise ValueError("perm size mismatch for cols")
    return [(np.asarray(cols[int(j)][0], dtype=np.int32, order="C"), np.asarray(cols[int(j)][1], dtype=np.float64, order="C")) for j in perm_a]


def _match_roots_by_overlap(*, prev_cols: Sequence[SparseCol], cols: Sequence[SparseCol]) -> List[int]:
    """Return a permutation that best matches `cols` to `prev_cols` by overlap.

    Both inputs must be L2-normalized columns in the same determinant basis.
    The returned `perm` is such that `cols_matched[i] = cols[perm[i]]` attempts
    to maximize |<prev_i | cols_matched_i>|.

    Uses a deterministic greedy assignment (nroots is small in practice).
    """

    n = int(len(cols))
    if int(len(prev_cols)) != n:
        return list(range(n))
    if n <= 1:
        return list(range(n))

    ov = np.zeros((n, n), dtype=np.float64)
    for i, (pi, pv) in enumerate(prev_cols):
        for j, (ci, cv) in enumerate(cols):
            ov[i, j] = float(abs(dot_sparse(pi, pv, ci, cv)))

    remaining = set(range(n))
    perm: list[int] = []
    for i in range(n):
        if not remaining:
            break
        # Pick the remaining column with maximum overlap to prev root i.
        best_j = max(remaining, key=lambda j: (ov[i, j], -j))
        perm.append(int(best_j))
        remaining.remove(best_j)

    if len(perm) != n:
        # Should not happen, but keep it safe/deterministic.
        perm.extend([int(j) for j in sorted(remaining)])
    return perm


def _det_subspace_topk_from_cols(*, cols: Sequence[SparseCol], k: int) -> np.ndarray | None:
    """Pick a deterministic subspace index set from sparse columns.

    Uses the maximum absolute coefficient across columns as a score per CSF
    index, then returns the top-k indices (sorted unique int32).
    """

    k = int(k)
    if k <= 0:
        return None

    best: Dict[int, float] = {}
    for idx, val in cols:
        idx_i64 = np.asarray(idx, dtype=np.int64).ravel()
        val_f = np.asarray(val, dtype=np.float64).ravel()
        if idx_i64.size == 0:
            continue
        for j, v in zip(idx_i64.tolist(), val_f.tolist()):
            a = abs(float(v))
            prev = best.get(int(j), 0.0)
            if a > prev:
                best[int(j)] = a

    if not best:
        return None

    keys = np.fromiter(best.keys(), dtype=np.int64)
    vals = np.fromiter(best.values(), dtype=np.float64)
    if int(keys.size) <= k:
        return np.asarray(np.sort(keys), dtype=np.int32)

    pos = np.argpartition(vals, -k)[-k:]
    return np.asarray(np.sort(keys[pos]), dtype=np.int32)


def _l1_norm(val: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(val, dtype=np.float64))))


def _scale_col(col: SparseCol, scale: float) -> SparseCol:
    idx, val = col
    if float(scale) == 1.0 or val.size == 0:
        return idx, val
    return np.asarray(idx, dtype=np.int32, order="C"), np.asarray(float(scale) * val, dtype=np.float64, order="C")


def _scale_cols(cols: Sequence[SparseCol], scales: Sequence[float]) -> List[SparseCol]:
    if len(cols) != len(scales):
        raise ValueError("cols/scales length mismatch")
    return [_scale_col(cols[k], float(scales[k])) for k in range(len(cols))]


def _safe_cond(A: np.ndarray) -> float:
    """Return cond(A) or NaN if it fails."""

    try:
        return float(np.linalg.cond(np.asarray(A, dtype=np.float64)))
    except Exception:
        return float("nan")


def _dense_from_cols(*, cols: Sequence[SparseCol], nrows: int) -> np.ndarray:
    """Build a dense (nrows, ncols) matrix from sparse columns (debug-only)."""

    ncols = int(len(cols))
    X = np.zeros((int(nrows), ncols), dtype=np.float64)
    for k, (idx_k, val_k) in enumerate(cols):
        idx_k = np.asarray(idx_k, dtype=np.int64).ravel()
        val_k = np.asarray(val_k, dtype=np.float64).ravel()
        if idx_k.size:
            X[idx_k, k] = val_k
    return X


def _l1_norm_right_multiply_no_phi(cols: Sequence[SparseCol], mat: np.ndarray) -> np.ndarray:
    """Compute columnwise ||X @ mat||_1 without Φ compression.

    This is used for Appendix-D D(τ) diagnostics and computation, where the
    paper defines norms on the uncompressed right-multiplied columns.
    """

    mat = np.asarray(mat, dtype=np.float64)
    n = int(len(cols))
    if mat.shape != (n, n):
        raise ValueError(f"mat has wrong shape: {mat.shape} (expected {(n, n)})")

    out = np.zeros(n, dtype=np.float64)
    for k in range(n):
        idx_chunks: List[np.ndarray] = []
        val_chunks: List[np.ndarray] = []
        for j in range(n):
            w = float(mat[j, k])
            if w == 0.0:
                continue
            idx_j, val_j = cols[j]
            if idx_j.size == 0:
                continue
            idx_chunks.append(np.asarray(idx_j, dtype=np.int32, order="C"))
            val_chunks.append(np.asarray(w * val_j, dtype=np.float64, order="C"))

        if not idx_chunks:
            out[k] = 0.0
            continue

        idx = np.concatenate(idx_chunks)
        val = np.concatenate(val_chunks)
        idx, val = coalesce_coo_i32_f64(idx, val)
        out[k] = _l1_norm(val)

    return out


def _phi_factory(
    *,
    kind: str,
    m: int,
    pivot: int,
    rng: np.random.Generator,
) -> Callable[[np.ndarray, np.ndarray], SparseCol]:
    """Return a Φ compression closure with signature (idx,val)->(idx,val)."""

    kind = str(kind).lower()
    if kind == "pivot_resample":
        return lambda idx, val: compress_phi_pivot_resample(idx, val, m=int(m), pivot=int(pivot), rng=rng)
    if kind == "pivotal":
        return lambda idx, val: compress_phi_pivotal(idx, val, m=int(m), rng=rng)
    raise ValueError("compression kind must be 'pivot_resample' or 'pivotal'")


def _resolve_per_root(value, *, k: int, nroots: int, name: str):
    if isinstance(value, (list, tuple)):
        if k >= len(value):
            raise ValueError(f"{name} sequence is shorter than number of roots (nroots={int(nroots)})")
        return value[k]
    return value


def apply_right_matrix_phi(
    cols: Sequence[SparseCol],
    mat: np.ndarray,
    *,
    phi: Callable[[np.ndarray, np.ndarray], SparseCol],
) -> List[SparseCol]:
    """Return Y = Φ(X @ mat) (column-wise), without any normalization."""

    mat = np.asarray(mat, dtype=np.float64)
    n = int(len(cols))
    if mat.shape != (n, n):
        raise ValueError(f"mat has wrong shape: {mat.shape} (expected {(n, n)})")

    out: List[SparseCol] = []
    for k in range(n):
        idx_chunks: List[np.ndarray] = []
        val_chunks: List[np.ndarray] = []
        for j in range(n):
            w = float(mat[j, k])
            if w == 0.0:
                continue
            idx_j, val_j = cols[j]
            if idx_j.size == 0:
                continue
            idx_chunks.append(np.asarray(idx_j, dtype=np.int32, order="C"))
            val_chunks.append(np.asarray(w * val_j, dtype=np.float64, order="C"))

        if not idx_chunks:
            raise ValueError("apply_right_matrix produced an empty column")

        idx = np.concatenate(idx_chunks)
        val = np.concatenate(val_chunks)
        idx, val = coalesce_coo_i32_f64(idx, val)
        idx, val = phi(idx, val)
        if idx.size == 0:
            raise ValueError("apply_right_matrix annihilated a column during Φ")
        out.append((idx, val))
    return out


def apply_right_matrix_no_phi(cols: Sequence[SparseCol], mat: np.ndarray) -> List[SparseCol]:
    """Return Y = X @ mat without Φ compression (still sparse-coalesced)."""

    mat = np.asarray(mat, dtype=np.float64)
    n = int(len(cols))
    if mat.shape != (n, n):
        raise ValueError(f"mat has wrong shape: {mat.shape} (expected {(n, n)})")

    out: List[SparseCol] = []
    for k in range(n):
        idx_chunks: List[np.ndarray] = []
        val_chunks: List[np.ndarray] = []
        for j in range(n):
            w = float(mat[j, k])
            if w == 0.0:
                continue
            idx_j, val_j = cols[j]
            if idx_j.size == 0:
                continue
            idx_chunks.append(np.asarray(idx_j, dtype=np.int32, order="C"))
            val_chunks.append(np.asarray(w * val_j, dtype=np.float64, order="C"))

        if not idx_chunks:
            raise ValueError("apply_right_matrix produced an empty column")

        idx = np.concatenate(idx_chunks)
        val = np.concatenate(val_chunks)
        idx, val = coalesce_coo_i32_f64(idx, val)
        if idx.size == 0:
            raise ValueError("apply_right_matrix annihilated a column")
        out.append((idx, val))
    return out


def _normalize_cols_l2(cols: Sequence[SparseCol]) -> List[SparseCol]:
    out: List[SparseCol] = []
    for k, (idx, val) in enumerate(cols):
        idx = np.asarray(idx, dtype=np.int32, order="C")
        val = np.asarray(val, dtype=np.float64, order="C")
        n2 = float(np.dot(val, val))
        if not (n2 > 0.0) or not np.isfinite(n2):
            raise ValueError(f"column {int(k)} has degenerate norm")
        out.append((idx, np.asarray(val / float(np.sqrt(n2)), dtype=np.float64, order="C")))
    return out


def build_utx(U_cols: Sequence[SparseCol], X_cols: Sequence[SparseCol]) -> np.ndarray:
    """Build S = U^T X from sparse columns."""

    n = int(len(U_cols))
    if len(X_cols) != n:
        raise ValueError("U_cols and X_cols must have same length")
    S = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        ui, uv = U_cols[i]
        for j in range(n):
            xj, xv = X_cols[j]
            S[i, j] = dot_sparse(ui, uv, xj, xv)
    return S


def _row_dot_with_sparse(
    row_idx: np.ndarray,
    row_val: np.ndarray,
    x_idx: np.ndarray,
    x_val: np.ndarray,
) -> float:
    """Compute (row · x) using searchsorted into sorted sparse x indices."""

    row_idx = np.asarray(row_idx, dtype=np.int32).ravel()
    row_val = np.asarray(row_val, dtype=np.float64).ravel()
    x_idx = np.asarray(x_idx, dtype=np.int32).ravel()
    x_val = np.asarray(x_val, dtype=np.float64).ravel()
    if row_idx.size == 0 or x_idx.size == 0:
        return 0.0
    pos = np.searchsorted(x_idx, row_idx)
    n = int(x_idx.size)
    in_bounds = pos < n
    if not np.any(in_bounds):
        return 0.0
    pos_ib = pos[in_bounds]
    row_ib = row_idx[in_bounds]
    hit = x_idx[pos_ib] == row_ib
    if not np.any(hit):
        return 0.0
    return float(np.dot(row_val[in_bounds][hit], x_val[pos_ib[hit]]))


def prepare_u_row_map(U_cols: Sequence[SparseCol]) -> Dict[int, List[Tuple[int, float]]]:
    """Precompute mapping p -> [(i, U[p,i]), ...] for all trial vectors."""

    p_to_terms: Dict[int, List[Tuple[int, float]]] = {}
    for i, (idx_i, val_i) in enumerate(U_cols):
        for p, up in zip(idx_i.tolist(), val_i.tolist()):
            if up == 0.0:
                continue
            lst = p_to_terms.get(int(p))
            if lst is None:
                p_to_terms[int(p)] = [(int(i), float(up))]
            else:
                lst.append((int(i), float(up)))
    return p_to_terms


def build_uthx_row_oracle(
    drt: DRT,
    h1e: np.ndarray,
    eri,
    *,
    p_to_terms: Mapping[int, Sequence[Tuple[int, float]]],
    X_cols: Sequence[SparseCol],
    max_out: int = 200_000,
    state_cache: DRTStateCache | None = None,
) -> np.ndarray:
    """Build K = U^T H X using deterministic row-oracle evaluations."""

    n = int(len(X_cols))
    K = np.zeros((n, n), dtype=np.float64)
    for p, terms in p_to_terms.items():
        row_idx, row_val = connected_row_sparse(
            drt,
            h1e,
            eri,
            int(p),
            max_out=int(max_out),
            state_cache=state_cache,
        )

        hx_p = np.zeros(n, dtype=np.float64)
        for k in range(n):
            x_idx, x_val = X_cols[k]
            hx_p[k] = _row_dot_with_sparse(row_idx, row_val, x_idx, x_val)

        for i, up in terms:
            K[int(i), :] += float(up) * hx_p

    return K


def _update_N_diag(
    *,
    l1_prev: np.ndarray,
    l1_curr: np.ndarray,
    N_prev: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Appendix-D diagonal update based on 1-norm ratios."""

    alpha = float(alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must satisfy 0<alpha<=1")
    l1_prev = np.asarray(l1_prev, dtype=np.float64).ravel()
    l1_curr = np.asarray(l1_curr, dtype=np.float64).ravel()
    N_prev = np.asarray(N_prev, dtype=np.float64).ravel()
    if l1_prev.shape != l1_curr.shape or l1_prev.shape != N_prev.shape:
        raise ValueError("shape mismatch for N update")

    ratio = np.ones_like(l1_curr)
    mask = l1_prev > 0.0
    ratio[mask] = l1_curr[mask] / l1_prev[mask]
    ratio = np.maximum(ratio, 1e-300)

    return (ratio**alpha) * (N_prev ** (1.0 - alpha))


def _generalized_eigs(K: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve K W = S W Λ for small matrices using NumPy."""

    K = np.asarray(K, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    if K.shape != S.shape or K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K and S must be same-shape square matrices")

    # In exact arithmetic (in the long-time/low-noise limit) the eigenvalues are real,
    # but at finite iteration count the (K,S) estimates can be noisy/non-normal.
    # Avoid hard-failing on small imaginary parts so the algorithm remains usable.
    A = np.linalg.solve(S, K)
    w, v = np.linalg.eig(A)
    if np.max(np.abs(np.imag(w))) > 1e-6:
        # Fall back to the symmetric part to obtain real-valued diagnostics.
        A_sym = 0.5 * (A + A.T)
        w, v = np.linalg.eigh(A_sym)
    w = np.real(w)
    v = np.real(v)
    order = np.argsort(w)
    return np.asarray(w[order], dtype=np.float64), np.asarray(v[:, order], dtype=np.float64)


def _delta_ortho_preN(
    *,
    it: int,
    U_cols: Sequence[SparseCol],
    X_cols: Sequence[SparseCol],
    Y_cols: Sequence[SparseCol],
    phi: Callable[[np.ndarray, np.ndarray], SparseCol],
    delta_ortho_basis: str,
    debug: bool,
    debug_dense_checks: bool,
    ncsf: int,
) -> List[SparseCol]:
    """Apply Appendix-D Δ-step mixing and D(τ) correction (pre-N scaling).

    Notes
    -----
    This routine returns columns corresponding to:
      preN = Φ(Y R^{-1}) D^{-1}
    where N^{-1} is applied by the caller after updating N(τ).
    """

    if debug:
        S_X = build_utx(U_cols, X_cols)
        S_Y = build_utx(U_cols, Y_cols)
        nSx = float(np.linalg.norm(S_X))
        nSy = float(np.linalg.norm(S_Y))
        cSx = _safe_cond(S_X)
        cSy = _safe_cond(S_Y)

        _Qx, Rx = np.linalg.qr(S_X)
        _Qy, Ry = np.linalg.qr(S_Y)
        dx = np.abs(np.diag(Rx))
        dy = np.abs(np.diag(Ry))
        rmin_x = float(np.min(dx)) if dx.size else float("nan")
        rmax_x = float(np.max(dx)) if dx.size else float("nan")
        rmin_y = float(np.min(dy)) if dy.size else float("nan")
        rmax_y = float(np.max(dy)) if dy.size else float("nan")

        print(
            f"[rsi dbg] it={int(it)}  ||U^T X||_F={nSx:.3e}  cond(U^T X)={cSx:.3e}  "
            f"min|diag(Rx)|={rmin_x:.3e}  max|diag(Rx)|={rmax_x:.3e}"
        )
        print(
            f"[rsi dbg] it={int(it)}  ||U^T Y||_F={nSy:.3e}  cond(U^T Y)={cSy:.3e}  "
            f"min|diag(Ry)|={rmin_y:.3e}  max|diag(Ry)|={rmax_y:.3e}"
        )

    basis = str(delta_ortho_basis).lower()
    if basis not in ("x", "y"):
        raise ValueError("delta_ortho_basis must be 'x' or 'y'")

    # Choose which iterate enters the QR construction.
    B_cols = X_cols if basis == "x" else Y_cols

    S = build_utx(U_cols, B_cols)
    _Q, R = np.linalg.qr(S)
    Rinv = np.linalg.inv(R)
    Z_cols = apply_right_matrix_phi(Y_cols, Rinv, phi=phi)

    denom_l1 = np.array([_l1_norm(v) for _, v in B_cols], dtype=np.float64)
    num_l1 = _l1_norm_right_multiply_no_phi(B_cols, Rinv)

    D_diag = np.ones(int(len(Y_cols)), dtype=np.float64)
    mask = denom_l1 > 0.0
    D_diag[mask] = num_l1[mask] / denom_l1[mask]
    D_diag = np.maximum(D_diag, 1e-300)

    preN_cols = _scale_cols(Z_cols, 1.0 / D_diag)

    if debug:
        dR = np.abs(np.diag(R))
        rmin = float(np.min(dR)) if dR.size else float("nan")
        rmax = float(np.max(dR)) if dR.size else float("nan")
        dmin = float(np.min(D_diag)) if D_diag.size else float("nan")
        dmax = float(np.max(D_diag)) if D_diag.size else float("nan")
        l1_min = float(np.min(denom_l1)) if denom_l1.size else float("nan")
        l1_max = float(np.max(denom_l1)) if denom_l1.size else float("nan")

        print(
            f"[rsi dbg] it={int(it)}  delta_ortho_basis={basis}  min|diag(R)|={rmin:.3e}  max|diag(R)|={rmax:.3e}  "
            f"D[min,max]=[{dmin:.3e},{dmax:.3e}]  "
            f"||basis||_1[min,max]=[{l1_min:.3e},{l1_max:.3e}]"
        )

        if debug_dense_checks:
            ncsf_i = int(ncsf)
            if ncsf_i <= 0 or ncsf_i > 50_000:
                print(f"[rsi dbg] it={int(it)}  dense checks skipped (ncsf={ncsf_i})")
            else:
                Xd = _dense_from_cols(cols=X_cols, nrows=ncsf_i)
                Yd = _dense_from_cols(cols=Y_cols, nrows=ncsf_i)

                # Compare paper-style (using U^T X) vs current (using U^T Y) D definitions.
                Sx = build_utx(U_cols, X_cols)
                _Qx2, Rx2 = np.linalg.qr(Sx)
                Rinv_x = np.linalg.inv(Rx2)
                Xr = Xd @ Rinv_x
                l1x = np.sum(np.abs(Xd), axis=0)
                l1xr = np.sum(np.abs(Xr), axis=0)
                D_x = np.ones_like(l1x)
                mx = l1x > 0.0
                D_x[mx] = l1xr[mx] / l1x[mx]

                Sy = build_utx(U_cols, Y_cols)
                _Qy2, Ry2 = np.linalg.qr(Sy)
                Rinv_y = np.linalg.inv(Ry2)
                Yr = Yd @ Rinv_y
                l1y = np.sum(np.abs(Yd), axis=0)
                l1yr = np.sum(np.abs(Yr), axis=0)
                D_y = np.ones_like(l1y)
                my = l1y > 0.0
                D_y[my] = l1yr[my] / l1y[my]

                # Report ranges and relative disagreement.
                Dx_min = float(np.min(D_x)) if D_x.size else float("nan")
                Dx_max = float(np.max(D_x)) if D_x.size else float("nan")
                Dy_min = float(np.min(D_y)) if D_y.size else float("nan")
                Dy_max = float(np.max(D_y)) if D_y.size else float("nan")
                rel = np.zeros_like(D_x)
                both = (D_x != 0.0) & np.isfinite(D_x) & np.isfinite(D_y)
                rel[both] = np.abs(D_y[both] - D_x[both]) / np.maximum(np.abs(D_x[both]), 1e-300)
                rel_max = float(np.max(rel[both])) if np.any(both) else float("nan")

                print(
                    f"[rsi dbg] it={int(it)}  D_from(U^T X)[min,max]=[{Dx_min:.3e},{Dx_max:.3e}]  "
                    f"D_from(U^T Y)[min,max]=[{Dy_min:.3e},{Dy_max:.3e}]  max_rel_diff={rel_max:.3e}"
                )

    return preN_cols


def _orthonormalize_trial_vectors_l2(U_cols: Sequence[SparseCol]) -> List[SparseCol]:
    """Deterministic modified Gram-Schmidt on trial vectors without Φ."""

    out: List[SparseCol] = []
    for k, (idx_k, val_k) in enumerate(U_cols):
        idx_k, val_k = coalesce_coo_i32_f64(idx_k, val_k)
        if idx_k.size == 0:
            raise ValueError(f"trial vector {k} is empty")

        for idx_i, val_i in out:
            ov = dot_sparse(idx_i, val_i, idx_k, val_k)
            if ov != 0.0:
                idx_k, val_k = axpy_sparse(idx_i, val_i, idx_k, val_k, alpha=-ov)

        n2 = float(np.dot(val_k, val_k))
        if n2 <= 0.0:
            raise ValueError(f"trial vector {k} has zero norm")
        val_k = np.asarray(val_k / float(np.sqrt(n2)), dtype=np.float64, order="C")
        out.append((np.asarray(idx_k, dtype=np.int32, order="C"), val_k))
    return out


def run_fcifri_rsi(
    drt: DRT,
    h1e: np.ndarray,
    eri,
    *,
    U0: Sequence[SparseCol],
    m: int,
    eps: float,
    niter: int,
    nspawn_one: int,
    nspawn_two: int,
    seed: int,
    omp_threads: int | None = None,
    pivot: int = 256,
    initiator_na: float = 0.0,
    alpha: float = 0.5,
    delta_ortho: int = 50,
    eval_stride: int = 10,
    burn_in: int = 50,
    compression: str = "pivotal",
    projector_compressor: Callable[..., tuple[np.ndarray, np.ndarray]] | Sequence[Callable[..., tuple[np.ndarray, np.ndarray]]] | None = None,
    projector_spawner: Callable[..., tuple[np.ndarray, np.ndarray]] | Sequence[Callable[..., tuple[np.ndarray, np.ndarray]]] | None = None,
    projector_spawner_kwargs: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
    backend: str = "stochastic",
    use_state_cache: bool = True,
    max_out_oracle: int = 200_000,
    iteration_hook: Callable[..., None] | None = None,
    eval_hook: RSIEvalHook | None = None,
    eval_hook_ritz: bool = True,
    root_tracking: str = "energy",
    projector_shift: str = "none",
    projector_shift_beta: float = 1.0,
    restart_from_ritz: bool = False,
    restart_orthonormalize: bool = False,
    restart_min_it: int = 0,
    restart_min_nsample: int = 10,
    restart_stride: int = 1,
    eval_k_nseeds: int = 1,
    eval_k_symmetrize: bool = True,
    cuda_det_nparent: int = 0,
    cuda_det_max_out: int = 200_000,
    orthonormalize_U: bool = True,
    delta_ortho_basis: str = "x",
    debug: bool = False,
    debug_dense_checks: bool = False,
) -> FCIFRIRSIResult:
    """Multi-root FCI-FRI excited states via RSI subspace iteration."""

    m = int(m)
    niter = int(niter)
    nroots = int(len(U0))
    if nroots < 1:
        raise ValueError("U0 must contain at least one trial vector")
    if m < 1:
        raise ValueError("m must be >= 1")
    if niter < 0:
        raise ValueError("niter must be >= 0")
    delta_ortho = int(delta_ortho)
    if delta_ortho < 1:
        raise ValueError("delta_ortho must be >= 1")
    eval_stride = int(eval_stride)
    if eval_stride < 1:
        raise ValueError("eval_stride must be >= 1")
    burn_in = int(burn_in)
    if burn_in < 0:
        raise ValueError("burn_in must be >= 0")
    delta_ortho_basis = str(delta_ortho_basis).lower()
    if delta_ortho_basis not in ("x", "y"):
        raise ValueError("delta_ortho_basis must be 'x' or 'y'")

    root_tracking = str(root_tracking).lower()
    if root_tracking not in ("energy", "overlap"):
        raise ValueError("root_tracking must be 'energy' or 'overlap'")

    projector_shift = str(projector_shift).lower()
    if projector_shift not in ("none", "eval"):
        raise ValueError("projector_shift must be 'none' or 'eval'")
    projector_shift_beta = float(projector_shift_beta)
    if not (0.0 < projector_shift_beta <= 1.0):
        raise ValueError("projector_shift_beta must satisfy 0<beta<=1")
    restart_from_ritz = bool(restart_from_ritz)
    restart_orthonormalize = bool(restart_orthonormalize)
    restart_min_it = int(restart_min_it)
    if restart_min_it < 0:
        raise ValueError("restart_min_it must be >= 0")
    restart_min_nsample = int(restart_min_nsample)
    if restart_min_nsample < 0:
        raise ValueError("restart_min_nsample must be >= 0")
    restart_stride = int(restart_stride)
    if restart_stride < 1:
        raise ValueError("restart_stride must be >= 1")

    eval_k_nseeds = int(eval_k_nseeds)
    if eval_k_nseeds < 1:
        raise ValueError("eval_k_nseeds must be >= 1")
    eval_k_symmetrize = bool(eval_k_symmetrize)

    cuda_det_nparent = int(cuda_det_nparent)
    if cuda_det_nparent < 0:
        raise ValueError("cuda_det_nparent must be >= 0")
    cuda_det_max_out = int(cuda_det_max_out)
    if cuda_det_max_out < 1:
        raise ValueError("cuda_det_max_out must be >= 1")

    backend = str(backend).lower()
    if backend not in ("stochastic", "exact", "cuda"):
        raise ValueError("backend must be 'stochastic', 'exact', or 'cuda'")

    if backend == "cuda" and (projector_spawner is not None or projector_spawner_kwargs is not None):
        warnings.warn(
            "backend='cuda' does not support custom spawner hooks yet; falling back to backend='stochastic'.",
            stacklevel=2,
        )
        backend = "stochastic"

    if backend == "cuda" and str(compression).lower() != "pivot_resample":
        raise ValueError("backend='cuda' currently requires compression='pivot_resample' (CUDA pivotal Φ not implemented)")

    if backend == "exact" and (projector_spawner is not None or projector_spawner_kwargs is not None):
        raise ValueError("projector_spawner/projector_spawner_kwargs are only used in backend='stochastic'")

    rng = np.random.default_rng(int(seed))
    maybe_set_openmp_threads(omp_threads)
    state_cache = get_state_cache(drt) if bool(use_state_cache) else None

    phi = _phi_factory(kind=str(compression), m=m, pivot=int(pivot), rng=rng)

    def default_projector_compressor(idx: np.ndarray, val: np.ndarray, **_kwargs) -> tuple[np.ndarray, np.ndarray]:
        return phi(idx, val)

    U_cols: List[SparseCol] = []
    for k, (idx_k, val_k) in enumerate(U0):
        idx_k, val_k = coalesce_coo_i32_f64(idx_k, val_k)
        if idx_k.size == 0:
            raise ValueError(f"trial vector {k} is empty")
        idx_k, val_k = phi(idx_k, val_k)
        if idx_k.size > int(m):
            raise RuntimeError("Φ(U) produced nnz>m (should not happen)")
        U_cols.append((idx_k, val_k))

    if orthonormalize_U:
        U_cols = _orthonormalize_trial_vectors_l2(U_cols)

    X_cols: List[SparseCol] = [(idx.copy(), val.copy()) for (idx, val) in U_cols]

    l1_x = np.array([_l1_norm(v) for _, v in X_cols], dtype=np.float64)
    N_diag = np.ones(nroots, dtype=np.float64)

    n_eval = (niter // eval_stride) + 1
    it_hist = np.zeros(n_eval, dtype=np.int32)
    e_inst = np.full((n_eval, nroots), np.nan, dtype=np.float64)
    e_avg = np.full((n_eval, nroots), np.nan, dtype=np.float64)
    S_sum = np.zeros((nroots, nroots), dtype=np.float64)
    K_sum = np.zeros((nroots, nroots), dtype=np.float64)
    nsample = 0
    prev_ritz_cols: List[SparseCol] | None = None
    shift_e: float | None = None

    def _restart_from_ritz_cols(ritz: Sequence[SparseCol]) -> List[SparseCol]:
        cols = _normalize_cols_l2([phi(idx, val) for (idx, val) in ritz])
        if restart_orthonormalize:
            cols = _orthonormalize_trial_vectors_l2(cols)
            cols = _normalize_cols_l2([phi(idx, val) for (idx, val) in cols])
        for k, (idx_k, _val_k) in enumerate(cols):
            if int(idx_k.size) > int(m):
                raise RuntimeError(f"restart produced column {k} nnz={int(idx_k.size)} (>m={int(m)})")
        return cols

    def _call_iteration_hook(*, it: int, x_cols: Sequence[SparseCol], y_cols: Sequence[SparseCol] | None) -> None:
        if iteration_hook is None:
            return
        x_copy = [(idx.copy(), val.copy()) for (idx, val) in x_cols]
        y_copy = None if y_cols is None else [(idx.copy(), val.copy()) for (idx, val) in y_cols]
        iteration_hook(
            it=int(it),
            x_cols=x_copy,
            y_cols=y_copy,
            u_cols=[(idx.copy(), val.copy()) for (idx, val) in U_cols],
            backend=str(backend),
        )

    def _record_eval(
        it: int, pos: int, *, S: np.ndarray, K: np.ndarray, X_cols_snap: Sequence[SparseCol] | None
    ) -> List[SparseCol] | None:
        nonlocal nsample, S_sum, K_sum
        it_hist[pos] = np.int32(it)
        w_inst, v_inst = _generalized_eigs(K, S)
        e_inst[pos] = w_inst

        if debug:
            cS = _safe_cond(S)
            try:
                A_inst = np.linalg.solve(S, K)
                wA = np.linalg.eigvals(A_inst)
                max_imag_inst = float(np.max(np.abs(np.imag(wA)))) if wA.size else 0.0
            except Exception:
                max_imag_inst = float("nan")
            print(f"[rsi dbg] eval it={int(it)}  cond(S)={cS:.3e}  max|Im(eig(S^-1 K))|={max_imag_inst:.3e}")

        if it >= burn_in:
            S_sum = S_sum + S
            K_sum = K_sum + K
            nsample += 1

        w_use = np.asarray(w_inst, dtype=np.float64)
        v_use = np.asarray(v_inst, dtype=np.float64)
        basis = "inst"
        if nsample > 0:
            S_avg = S_sum / float(nsample)
            K_avg = K_sum / float(nsample)
            w_avg, v_avg = _generalized_eigs(K_avg, S_avg)
            e_avg[pos] = w_avg
            w_use = np.asarray(w_avg, dtype=np.float64)
            v_use = np.asarray(v_avg, dtype=np.float64)
            basis = "avg"
            if debug:
                cS_avg = _safe_cond(S_avg)
                try:
                    A_avg = np.linalg.solve(S_avg, K_avg)
                    wA_avg = np.linalg.eigvals(A_avg)
                    max_imag_avg = float(np.max(np.abs(np.imag(wA_avg)))) if wA_avg.size else 0.0
                except Exception:
                    max_imag_avg = float("nan")
                print(
                    f"[rsi dbg] eval it={int(it)}  nsample={int(nsample)}  cond(<S>)={cS_avg:.3e}  "
                    f"max|Im(eig(<S>^-1 <K>))|={max_imag_avg:.3e}"
                )

        nonlocal prev_ritz_cols
        ritz_cols: List[SparseCol] | None = None
        need_ritz = bool(eval_hook is not None and eval_hook_ritz) or (root_tracking == "overlap")
        if need_ritz and X_cols_snap is not None:
            ritz_cols = _normalize_cols_l2(apply_right_matrix_no_phi(list(X_cols_snap), v_use))

        if root_tracking == "overlap" and ritz_cols is not None:
            if prev_ritz_cols is not None:
                perm = _match_roots_by_overlap(prev_cols=prev_ritz_cols, cols=ritz_cols)
                w_use = np.asarray(w_use[list(perm)], dtype=np.float64)
                v_use = _permute_vecs_cols(v_use, perm)
                ritz_cols = _permute_cols(ritz_cols, perm)

                # Fix signs deterministically relative to previous Ritz columns.
                fixed_cols: list[SparseCol] = []
                for k in range(int(len(ritz_cols))):
                    idx_k, val_k = ritz_cols[k]
                    s = float(dot_sparse(prev_ritz_cols[k][0], prev_ritz_cols[k][1], idx_k, val_k))
                    if s < 0.0:
                        val_k = np.asarray(-val_k, dtype=np.float64, order="C")
                        v_use[:, k] = -v_use[:, k]
                    fixed_cols.append((idx_k, val_k))
                ritz_cols = fixed_cols
            prev_ritz_cols = [(idx.copy(), val.copy()) for (idx, val) in ritz_cols]

        # Keep the stored energies aligned with the (optionally tracked) ordering used
        # for hooks/diagnostics. The "unused" basis (inst vs avg) remains recorded
        # in its default energy-sorted order above.
        if basis == "inst":
            e_inst[pos] = np.asarray(w_use, dtype=np.float64)
        else:
            e_avg[pos] = np.asarray(w_use, dtype=np.float64)

        nonlocal shift_e
        if projector_shift != "none":
            w_shift = np.asarray(w_use, dtype=np.float64).ravel()
            ok = np.isfinite(w_shift)
            if np.any(ok):
                target = float(np.mean(w_shift[ok]))
                if shift_e is None:
                    shift_e = float(target)
                else:
                    beta = float(projector_shift_beta)
                    shift_e = (1.0 - beta) * float(shift_e) + beta * float(target)

        if eval_hook is not None:
            x_cols_list = [] if X_cols_snap is None else [(idx.copy(), val.copy()) for (idx, val) in X_cols_snap]
            eval_hook(int(it), np.asarray(w_use, dtype=np.float64), np.asarray(v_use, dtype=np.float64), x_cols_list, ritz_cols, str(basis), int(nsample))
        return ritz_cols

    if backend != "cuda":
        p_to_terms = prepare_u_row_map(U_cols)

        def do_eval_cpu(it: int, pos: int) -> List[SparseCol] | None:
            S = build_utx(U_cols, X_cols)
            K = build_uthx_row_oracle(
                drt,
                h1e,
                eri,
                p_to_terms=p_to_terms,
                X_cols=X_cols,
                max_out=int(max_out_oracle),
                state_cache=state_cache,
            )
            return _record_eval(it, pos, S=S, K=K, X_cols_snap=list(X_cols))

    eval_pos = 0

    if backend == "cuda":
        try:
            import cupy as cp  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"backend='cuda' requires cupy: {e}") from e

        try:
            from .cuda_backend import (
                cuda_block_build_sk_uthx_stochastic_ws,
                cuda_block_projector_step_hamiltonian_ws,
                make_cuda_block_projector_context,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"failed to import CUDA QMC backend: {e}") from e

        det_idx = None
        if int(cuda_det_nparent) > 0:
            det_idx = _det_subspace_topk_from_cols(cols=U_cols, k=int(cuda_det_nparent))

        ctx = make_cuda_block_projector_context(
            drt,
            h1e,
            eri,
            nroots=int(nroots),
            m=int(m),
            pivot=int(pivot),
            nspawn_one=int(nspawn_one),
            nspawn_two=int(nspawn_two),
            det_idx=det_idx,
            det_max_out=int(cuda_det_max_out),
        )
        try:
            ctx.set_cols(X_cols)

            def do_eval_cuda(it: int, pos: int) -> List[SparseCol] | None:
                S = None
                K_sum = None
                for _t in range(int(eval_k_nseeds)):
                    seeds_eval = rng.integers(0, np.iinfo(np.int64).max, size=nroots, dtype=np.int64)
                    S_t, K_t = cuda_block_build_sk_uthx_stochastic_ws(
                        ctx,
                        u_cols=U_cols,
                        seeds_spawn=seeds_eval,
                        eps=1.0,
                        initiator_t=0.0,
                        sync=True,
                    )
                    if S is None:
                        S = np.asarray(S_t, dtype=np.float64)
                    if K_sum is None:
                        K_sum = np.asarray(K_t, dtype=np.float64)
                    else:
                        K_sum = K_sum + np.asarray(K_t, dtype=np.float64)
                assert S is not None and K_sum is not None
                K = K_sum / float(eval_k_nseeds)
                if eval_k_symmetrize:
                    K = 0.5 * (K + K.T)
                return _record_eval(it, pos, S=S, K=K, X_cols_snap=list(X_cols))

            _call_iteration_hook(it=0, x_cols=X_cols, y_cols=None)
            ritz0 = do_eval_cuda(0, eval_pos)
            eval_pos += 1
            if restart_from_ritz and ritz0 is not None and int(0) >= int(restart_min_it) and int(nsample) >= int(restart_min_nsample):
                X_cols = _restart_from_ritz_cols(ritz0)
                ctx.set_cols(X_cols)

            for it in range(1, niter + 1):
                l1_curr_x = np.array([_l1_norm(v) for _, v in X_cols], dtype=np.float64)
                initiator_t = np.array(
                    [initiator_threshold(l1_norm=float(l1_curr_x[k]), m=int(m), na=float(initiator_na)) for k in range(nroots)],
                    dtype=np.float64,
                )

                seed_spawn = rng.integers(0, np.iinfo(np.int64).max, size=nroots, dtype=np.int64)
                seed_phi = rng.integers(0, np.iinfo(np.int64).max, size=nroots, dtype=np.int64)
                scale_identity = 1.0
                if projector_shift != "none" and shift_e is not None:
                    scale_identity = 1.0 + float(eps) * float(shift_e)
                cuda_block_projector_step_hamiltonian_ws(
                    ctx,
                    eps=float(eps),
                    initiator_t=initiator_t,
                    seed_spawn=seed_spawn,
                    seed_phi=seed_phi,
                    scale_identity=scale_identity,
                    sync=True,
                    compressor=projector_compressor,
                )

                Y_cols = ctx.get_cols()

                if it % delta_ortho == 0:
                    preN_cols = _delta_ortho_preN(
                        it=it,
                        U_cols=U_cols,
                        X_cols=X_cols,
                        Y_cols=Y_cols,
                        phi=phi,
                        delta_ortho_basis=str(delta_ortho_basis),
                        debug=bool(debug),
                        debug_dense_checks=bool(debug_dense_checks),
                        ncsf=int(drt.ncsf),
                    )
                else:
                    preN_cols = Y_cols

                X_next = _scale_cols(preN_cols, 1.0 / N_diag)
                l1_next_x = np.array([_l1_norm(v) for _, v in X_next], dtype=np.float64)
                N_diag = _update_N_diag(l1_prev=l1_x, l1_curr=l1_next_x, N_prev=N_diag, alpha=float(alpha))
                X_cols = X_next
                l1_x = l1_next_x

                ctx.set_cols(X_cols)
                _call_iteration_hook(it=it, x_cols=X_cols, y_cols=Y_cols)

                if it % eval_stride == 0:
                    ritz = do_eval_cuda(it, eval_pos)
                    eval_pos += 1
                    do_restart = (
                        restart_from_ritz
                        and ritz is not None
                        and int(it) >= int(restart_min_it)
                        and int(nsample) >= int(restart_min_nsample)
                        and (int(eval_pos) % int(restart_stride) == 0)
                    )
                    if do_restart:
                        X_cols = _restart_from_ritz_cols(ritz)
                        ctx.set_cols(X_cols)

            X_cols = ctx.get_cols()
        finally:
            ctx.release()
    else:
        _call_iteration_hook(it=0, x_cols=X_cols, y_cols=None)
        ritz0 = do_eval_cpu(0, eval_pos)
        eval_pos += 1
        if restart_from_ritz and ritz0 is not None and int(0) >= int(restart_min_it) and int(nsample) >= int(restart_min_nsample):
            X_cols = _restart_from_ritz_cols(ritz0)

        for it in range(1, niter + 1):
            Y_cols: List[SparseCol] = []
            for k in range(nroots):
                idx_k, val_k = X_cols[k]
                comp_k = _resolve_per_root(projector_compressor, k=k, nroots=nroots, name="projector_compressor")
                sp_k = _resolve_per_root(projector_spawner, k=k, nroots=nroots, name="projector_spawner")
                sp_kw_k = _resolve_per_root(
                    projector_spawner_kwargs,
                    k=k,
                    nroots=nroots,
                    name="projector_spawner_kwargs",
                )
                idx_y, val_y = projector_step(
                    drt,
                    h1e,
                    eri,
                    idx_k,
                    val_k,
                    eps=float(eps),
                    nspawn_one=int(nspawn_one),
                    nspawn_two=int(nspawn_two),
                    rng=rng,
                    m=int(m),
                    pivot=int(pivot),
                    initiator_na=float(initiator_na),
                    scale_identity=(1.0 + float(eps) * float(shift_e)) if (projector_shift != "none" and shift_e is not None) else 1.0,
                    state_cache=state_cache,
                    backend=backend,
                    compressor=default_projector_compressor if comp_k is None else comp_k,
                    spawner=sp_k,
                    spawner_kwargs=sp_kw_k,
                )
                Y_cols.append((idx_y, val_y))

            if it % delta_ortho == 0:
                preN_cols = _delta_ortho_preN(
                    it=it,
                    U_cols=U_cols,
                    X_cols=X_cols,
                    Y_cols=Y_cols,
                    phi=phi,
                    delta_ortho_basis=str(delta_ortho_basis),
                    debug=bool(debug),
                    debug_dense_checks=bool(debug_dense_checks),
                    ncsf=int(drt.ncsf),
                )
            else:
                preN_cols = Y_cols

            X_next = _scale_cols(preN_cols, 1.0 / N_diag)
            l1_next_x = np.array([_l1_norm(v) for _, v in X_next], dtype=np.float64)
            N_diag = _update_N_diag(l1_prev=l1_x, l1_curr=l1_next_x, N_prev=N_diag, alpha=float(alpha))
            X_cols = X_next
            l1_x = l1_next_x
            _call_iteration_hook(it=it, x_cols=X_cols, y_cols=Y_cols)

            if it % eval_stride == 0:
                ritz = do_eval_cpu(it, eval_pos)
                eval_pos += 1
                do_restart = (
                    restart_from_ritz
                    and ritz is not None
                    and int(it) >= int(restart_min_it)
                    and int(nsample) >= int(restart_min_nsample)
                    and (int(eval_pos) % int(restart_stride) == 0)
                )
                if do_restart:
                    X_cols = _restart_from_ritz_cols(ritz)

    if nsample > 0:
        S_avg = S_sum / float(nsample)
        K_avg = K_sum / float(nsample)
    else:
        S_avg = np.zeros((nroots, nroots), dtype=np.float64)
        K_avg = np.zeros((nroots, nroots), dtype=np.float64)

    return FCIFRIRSIResult(
        idx=[c[0] for c in X_cols],
        val=[c[1] for c in X_cols],
        iters=it_hist[:eval_pos],
        energies_inst=e_inst[:eval_pos],
        energies_avg=e_avg[:eval_pos],
        s_avg=S_avg,
        k_avg=K_avg,
        nsample=int(nsample),
        backend=backend,
    )
