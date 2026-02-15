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

    max_cycle = int(max_cycle)
    max_space = int(max_space)
    if max_cycle < 1:
        raise ValueError("max_cycle must be >= 1")
    if max_space < 2:
        raise ValueError("max_space must be >= 2")

    x0 = np.asarray(x0, dtype=np.float64).ravel()
    if x0.size == 0:
        raise ValueError("x0 must be non-empty")

    s0 = overlap(x0)
    s_norm2 = float(np.dot(x0, np.asarray(s0, dtype=np.float64)))
    if s_norm2 <= 0.0:
        raise ValueError("initial vector has non-positive S-norm")
    x0 = x0 / np.sqrt(s_norm2)

    v_list: list[np.ndarray] = [x0]
    hv_list: list[np.ndarray] = []
    sv_list: list[np.ndarray] = []

    e = 0.0
    resid_norm = float("inf")
    converged = False

    for it in range(max_cycle):
        # Expand cached subspace products if needed.
        while len(hv_list) < len(v_list):
            v = v_list[len(hv_list)]
            hv_list.append(np.asarray(sigma(v), dtype=np.float64).ravel())
            sv_list.append(np.asarray(overlap(v), dtype=np.float64).ravel())

        v_mat = np.column_stack(v_list)  # (n, m)
        hv_mat = np.column_stack(hv_list)
        sv_mat = np.column_stack(sv_list)

        h_sub = _symmetrize(v_mat.T @ hv_mat)
        s_sub = _symmetrize(v_mat.T @ sv_mat)

        e, x_sub = _solve_gen_eigh_lowest(h=h_sub, s=s_sub, s_tol=s_tol)

        c = v_mat @ x_sub
        hc = hv_mat @ x_sub
        sc = sv_mat @ x_sub
        c_snorm2 = float(np.dot(c, sc))
        if c_snorm2 <= 0.0:
            raise np.linalg.LinAlgError("Ritz vector has non-positive S-norm")
        c_scale = 1.0 / np.sqrt(c_snorm2)
        c = c * c_scale
        hc = hc * c_scale
        sc = sc * c_scale

        r = hc - e * sc
        # Dimension-normalized residual: use RMS rather than 2-norm.
        # This makes convergence behavior more stable across problem sizes.
        resid_norm = float(np.linalg.norm(r)) / np.sqrt(float(r.size))
        if resid_norm < float(tol):
            converged = True
            return GeneralizedDavidsonResult(
                converged=True,
                e=float(e),
                x=np.ascontiguousarray(c, dtype=np.float64),
                niter=it + 1,
                residual_norm=float(resid_norm),
            )

        t = r if precond is None else np.asarray(precond(r, float(e)), dtype=np.float64).ravel()
        if t.size != x0.size:
            raise ValueError("precond returned a vector with wrong shape")

        # S-orthogonalize: enforce V^T S t = 0 using the cached sv_list.
        st = np.asarray(overlap(t), dtype=np.float64).ravel()
        proj = v_mat.T @ st
        t = t - v_mat @ proj

        st = np.asarray(overlap(t), dtype=np.float64).ravel()
        t_norm2 = float(np.dot(t, st))
        if t_norm2 <= float(s_tol):
            # If we can't add a new direction (linear dependency), restart from current Ritz vector.
            v_list = [c / np.sqrt(float(np.dot(c, sc)))]
            hv_list = []
            sv_list = []
            continue

        t = t / np.sqrt(t_norm2)

        v_list.append(t)
        if len(v_list) > max_space:
            # Thick-ish restart: keep the current Ritz vector and the most recent
            # correction direction (often much better than restarting from c alone).
            v_list = [c, t]
            hv_list = []
            sv_list = []

    while len(hv_list) < len(v_list):
        v = v_list[len(hv_list)]
        hv_list.append(np.asarray(sigma(v), dtype=np.float64).ravel())
        sv_list.append(np.asarray(overlap(v), dtype=np.float64).ravel())

    # Finalize with the last Ritz vector.
    v_mat = np.column_stack(v_list)
    hv_mat = np.column_stack(hv_list)
    sv_mat = np.column_stack(sv_list)
    h_sub = _symmetrize(v_mat.T @ hv_mat)
    s_sub = _symmetrize(v_mat.T @ sv_mat)
    e, x_sub = _solve_gen_eigh_lowest(h=h_sub, s=s_sub, s_tol=s_tol)
    c = v_mat @ x_sub
    sc = sv_mat @ x_sub
    c_snorm2 = float(np.dot(c, sc))
    if c_snorm2 > 0.0:
        c = c / np.sqrt(c_snorm2)
    return GeneralizedDavidsonResult(
        converged=bool(converged),
        e=float(e),
        x=np.ascontiguousarray(c, dtype=np.float64),
        niter=int(max_cycle),
        residual_norm=float(resid_norm),
    )
