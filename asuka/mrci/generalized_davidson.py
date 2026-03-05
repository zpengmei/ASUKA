from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Vec = np.ndarray
Matvec = Callable[[Vec], Vec]
Precond = Callable[[Vec, float], Vec]


@dataclass(frozen=True)
class GeneralizedDavidsonResult:
    """Result container for the generalized Davidson eigensolver.

    Attributes
    ----------
    converged : bool
        Whether the solver converged within the given tolerance.
    e : float
        The lowest eigenvalue found.
    x : np.ndarray
        The corresponding eigenvector, normalized such that x^T S x = 1.
    niter : int
        Number of iterations performed.
    residual_norm : float
        Norm of the residual vector (r = Hx - E Sx).
    """

    converged: bool
    e: float
    x: np.ndarray
    niter: int
    residual_norm: float


@dataclass(frozen=True)
class GeneralizedDavidsonResultMulti:
    """Result container for the multi-root generalized Davidson eigensolver."""

    converged: np.ndarray
    e: np.ndarray
    x: list[np.ndarray]
    niter: int
    residual_norm: np.ndarray


def _symmetrize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def _solve_gen_eigh_lowest(
    *,
    h: np.ndarray,
    s: np.ndarray,
    s_tol: float,
) -> tuple[float, np.ndarray]:
    """Solve the dense generalized eigenproblem in a subspace, return lowest root."""

    h = np.asarray(h, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if h.shape != s.shape or h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h and s must be square and have the same shape")

    # Robustly handle an ill-conditioned overlap by eigen-filtering S.
    s = _symmetrize(s)
    evals_s, evecs_s = np.linalg.eigh(s)
    keep = evals_s > float(s_tol)
    if not np.any(keep):
        raise np.linalg.LinAlgError("overlap matrix is numerically singular")

    u = evecs_s[:, keep]
    s_inv_sqrt = np.diag(1.0 / np.sqrt(evals_s[keep]))
    t = u @ s_inv_sqrt  # columns are S^{-1/2} basis vectors

    h_ortho = _symmetrize(t.T @ _symmetrize(h) @ t)
    evals, evecs = np.linalg.eigh(h_ortho)
    idx = int(np.argmin(evals))
    e0 = float(evals[idx])
    x0 = t @ evecs[:, idx]
    return e0, np.asarray(x0, dtype=np.float64)


def _solve_gen_eigh_lowest_n(
    *,
    h: np.ndarray,
    s: np.ndarray,
    s_tol: float,
    nroots: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the dense generalized eigenproblem in a subspace, return lowest roots."""

    h = np.asarray(h, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be >= 1")
    if h.shape != s.shape or h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h and s must be square and have the same shape")

    s = _symmetrize(s)
    evals_s, evecs_s = np.linalg.eigh(s)
    keep = evals_s > float(s_tol)
    if not np.any(keep):
        raise np.linalg.LinAlgError("overlap matrix is numerically singular")

    u = evecs_s[:, keep]
    s_inv_sqrt = np.diag(1.0 / np.sqrt(evals_s[keep]))
    t = u @ s_inv_sqrt

    h_ortho = _symmetrize(t.T @ _symmetrize(h) @ t)
    evals, evecs = np.linalg.eigh(h_ortho)
    nkeep = min(int(nroots), int(evals.size))
    if nkeep < 1:
        raise np.linalg.LinAlgError("no generalized eigenpairs available in subspace")
    e = np.asarray(evals[:nkeep], dtype=np.float64)
    x = np.asarray(t @ evecs[:, :nkeep], dtype=np.float64)
    return e, x


def generalized_davidson(
    sigma: Matvec,
    overlap: Matvec,
    x0: list[Vec],
    *,
    precond: Precond | None = None,
    tol: float = 1e-10,
    max_cycle: int = 100,
    max_space: int = 20,
    s_tol: float = 1e-12,
    nroots: int = 1,
) -> GeneralizedDavidsonResultMulti:
    """Solve the generalized eigenvalue problem Hc = ESc for the lowest roots."""

    max_cycle = int(max_cycle)
    max_space = int(max_space)
    nroots = int(nroots)
    if max_cycle < 1:
        raise ValueError("max_cycle must be >= 1")
    if nroots < 1:
        raise ValueError("nroots must be >= 1")
    if max_space < max(2, nroots):
        raise ValueError("max_space must be >= max(2, nroots)")
    if not x0:
        raise ValueError("x0 must not be empty")

    n = int(np.asarray(x0[0], dtype=np.float64).size)
    if n <= 0:
        raise ValueError("x0 vectors must be non-empty")

    v_list: list[np.ndarray] = []
    hv_list: list[np.ndarray] = []
    sv_list: list[np.ndarray] = []
    for guess in x0:
        vg = np.asarray(guess, dtype=np.float64).ravel()
        if int(vg.size) != n:
            raise ValueError("all initial guesses must have the same length")
        svg = np.asarray(overlap(vg), dtype=np.float64).ravel()
        snorm2 = float(np.dot(vg, svg))
        if snorm2 <= 0.0:
            continue
        vg = vg / np.sqrt(snorm2)
        if v_list:
            basis = np.column_stack(v_list)
            proj = basis.T @ np.asarray(overlap(vg), dtype=np.float64).ravel()
            vg = vg - basis @ proj
            svg = np.asarray(overlap(vg), dtype=np.float64).ravel()
            snorm2 = float(np.dot(vg, svg))
            if snorm2 <= float(s_tol):
                continue
            vg = vg / np.sqrt(snorm2)
        v_list.append(np.asarray(vg, dtype=np.float64))
        if len(v_list) >= nroots:
            break
    if len(v_list) < nroots:
        raise RuntimeError("failed to build enough linearly independent initial vectors")

    e = np.zeros((nroots,), dtype=np.float64)
    resid_norm = np.full((nroots,), float("inf"), dtype=np.float64)
    converged = np.zeros((nroots,), dtype=bool)
    c_mat = None

    for it in range(max_cycle):
        while len(hv_list) < len(v_list):
            v = v_list[len(hv_list)]
            hv_list.append(np.asarray(sigma(v), dtype=np.float64).ravel())
            sv_list.append(np.asarray(overlap(v), dtype=np.float64).ravel())

        v_mat = np.column_stack(v_list)
        hv_mat = np.column_stack(hv_list)
        sv_mat = np.column_stack(sv_list)

        h_sub = _symmetrize(v_mat.T @ hv_mat)
        s_sub = _symmetrize(v_mat.T @ sv_mat)
        e, x_sub = _solve_gen_eigh_lowest_n(h=h_sub, s=s_sub, s_tol=s_tol, nroots=nroots)

        c_mat = np.asarray(v_mat @ x_sub, dtype=np.float64)
        hc_mat = np.asarray(hv_mat @ x_sub, dtype=np.float64)
        sc_mat = np.asarray(sv_mat @ x_sub, dtype=np.float64)
        for root in range(int(c_mat.shape[1])):
            snorm2 = float(np.dot(c_mat[:, root], sc_mat[:, root]))
            if snorm2 <= 0.0:
                raise np.linalg.LinAlgError("Ritz vector has non-positive S-norm")
            scale = 1.0 / np.sqrt(snorm2)
            c_mat[:, root] *= scale
            hc_mat[:, root] *= scale
            sc_mat[:, root] *= scale

        r_mat = hc_mat - sc_mat * e[None, :]
        resid_norm = np.linalg.norm(r_mat, axis=0) / np.sqrt(float(n))
        converged = resid_norm < float(tol)
        if bool(np.all(converged)):
            xs = [np.ascontiguousarray(c_mat[:, i], dtype=np.float64) for i in range(int(c_mat.shape[1]))]
            return GeneralizedDavidsonResultMulti(
                converged=np.asarray(converged, dtype=bool),
                e=np.asarray(e, dtype=np.float64),
                x=xs,
                niter=it + 1,
                residual_norm=np.asarray(resid_norm, dtype=np.float64),
            )

        new_vecs: list[np.ndarray] = []
        for root in range(int(c_mat.shape[1])):
            if bool(converged[root]):
                continue
            t = np.asarray(r_mat[:, root], dtype=np.float64) if precond is None else np.asarray(
                precond(np.asarray(r_mat[:, root], dtype=np.float64), float(e[root])),
                dtype=np.float64,
            )
            if int(t.size) != n:
                raise ValueError("precond returned a vector with wrong shape")
            st = np.asarray(overlap(t), dtype=np.float64).ravel()
            proj = v_mat.T @ st
            t = t - v_mat @ proj
            if new_vecs:
                q = np.column_stack(new_vecs)
                sq = np.column_stack([np.asarray(overlap(v), dtype=np.float64).ravel() for v in new_vecs])
                proj_new = sq.T @ t
                t = t - q @ proj_new
            st = np.asarray(overlap(t), dtype=np.float64).ravel()
            t_norm2 = float(np.dot(t, st))
            if t_norm2 <= float(s_tol):
                continue
            new_vecs.append(np.asarray(t / np.sqrt(t_norm2), dtype=np.float64))

        if not new_vecs:
            if c_mat is not None:
                v_list = [np.asarray(c_mat[:, i], dtype=np.float64).copy() for i in range(int(c_mat.shape[1]))]
                hv_list = []
                sv_list = []
            continue

        if len(v_list) + len(new_vecs) > max_space:
            v_list = [np.asarray(c_mat[:, i], dtype=np.float64).copy() for i in range(int(c_mat.shape[1]))]
            hv_list = []
            sv_list = []
            continue

        v_list.extend(new_vecs)

    if c_mat is None:  # pragma: no cover
        raise RuntimeError("internal error: missing Ritz vectors")
    xs = [np.ascontiguousarray(c_mat[:, i], dtype=np.float64) for i in range(int(c_mat.shape[1]))]
    return GeneralizedDavidsonResultMulti(
        converged=np.asarray(converged, dtype=bool),
        e=np.asarray(e, dtype=np.float64),
        x=xs,
        niter=int(max_cycle),
        residual_norm=np.asarray(resid_norm, dtype=np.float64),
    )


def generalized_davidson1(
    sigma: Matvec,
    overlap: Matvec,
    x0: Vec,
    *,
    precond: Precond | None = None,
    tol: float = 1e-10,
    max_cycle: int = 100,
    max_space: int = 20,
    s_tol: float = 1e-12,
) -> GeneralizedDavidsonResult:
    """Solve the generalized eigenvalue problem Hc = ESc for the lowest root using Davidson's method.

    Parameters
    ----------
    sigma : Callable[[np.ndarray], np.ndarray]
        Callback function computing the matrix-vector product H @ x.
    overlap : Callable[[np.ndarray], np.ndarray]
        Callback function computing the matrix-vector product S @ x.
    x0 : np.ndarray
        Initial guess vector.
    precond : Callable[[np.ndarray, float], np.ndarray] | None, optional
        Preconditioner callback returning an update direction given residual `r` and
        current eigenvalue `e`. If None, the raw residual is used.
    tol : float, optional
        Convergence tolerance for the residual norm. Default is 1e-10.
    max_cycle : int, optional
        Maximum number of iterations. Default is 100.
    max_space : int, optional
        Maximum subspace size before restart. Default is 20.
    s_tol : float, optional
        Tolerance for linear dependency check in the subspace. Default is 1e-12.

    Returns
    -------
    GeneralizedDavidsonResult
        Result object containing the lowest eigenvalue, eigenvector, and convergence statistics.
    """

    res = generalized_davidson(
        sigma,
        overlap,
        [np.asarray(x0, dtype=np.float64).ravel()],
        precond=precond,
        tol=float(tol),
        max_cycle=int(max_cycle),
        max_space=int(max_space),
        s_tol=float(s_tol),
        nroots=1,
    )
    return GeneralizedDavidsonResult(
        converged=bool(np.asarray(res.converged, dtype=bool).ravel()[0]),
        e=float(np.asarray(res.e, dtype=np.float64).ravel()[0]),
        x=np.ascontiguousarray(res.x[0], dtype=np.float64),
        niter=int(res.niter),
        residual_norm=float(np.asarray(res.residual_norm, dtype=np.float64).ravel()[0]),
    )
