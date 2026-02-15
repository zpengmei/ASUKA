from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np


Vec = np.ndarray
AOP = Callable[[list[Vec]], list[Vec]]
Precond = Callable[[Vec, float, Vec], Vec]


@dataclass
class DavidsonResult:
    """Result of a Davidson eigenvalue computation.

    Attributes
    ----------
    converged : np.ndarray
        Boolean array of shape ``(nroots,)`` indicating convergence per root.
    e : np.ndarray
        float64 array of shape ``(nroots,)`` with eigenvalues.
    x : list of np.ndarray
        List of ``nroots`` eigenvectors, each a float64 array of length ``n``.
    niter : int
        Number of Davidson iterations performed.
    stats : dict or None
        Profiling statistics (if ``profile=True``), including
        ``hop_calls``, ``hop_time_s``, ``orth_time_s``, ``subspace_time_s``.
    """

    converged: np.ndarray  # (nroots,) bool
    e: np.ndarray  # (nroots,) float64
    x: list[np.ndarray]  # list of (n,) float64
    niter: int
    stats: dict[str, float] | None = None


def _as_f64_vec(x: Any, *, n: int | None = None) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if n is not None and int(v.size) != int(n):
        raise ValueError("vector has wrong size")
    return v


def _orthonormalize_one(v: np.ndarray, *, basis: np.ndarray | None, lindep: float) -> np.ndarray | None:
    if basis is not None and int(basis.shape[1]) > 0:
        v = v - basis @ (basis.T @ v)
    nrm = float(np.linalg.norm(v))
    if nrm <= float(lindep):
        return None
    return v / nrm


def _aop_block(aop: AOP, v: np.ndarray) -> np.ndarray:
    """Return W = A @ V for a Fortran-ordered (n, m) basis matrix V."""

    if v.ndim != 2:
        raise ValueError("V must be 2D")
    n, m = map(int, v.shape)
    if n <= 0 or m <= 0:
        raise ValueError("invalid V shape")

    xs = [np.asarray(v[:, i], dtype=np.float64, order="C") for i in range(m)]
    ys = aop(xs)
    if not isinstance(ys, list) or len(ys) != m:
        raise ValueError("aop returned an unexpected result (expected list of length m)")
    out = np.empty((n, m), dtype=np.float64, order="F")
    for i, y in enumerate(ys):
        out[:, i] = _as_f64_vec(y, n=n)
    return out


def _davidson_impl(
    aop: AOP,
    x0: list[Vec],
    precond: Precond,
    *,
    tol: float = 1e-10,
    lindep: float = 1e-14,
    max_cycle: int = 50,
    max_space: int = 12,
    nroots: int = 1,
    tol_residual: float | None = None,
    profile: bool = False,
    **_kwargs: Any,
) -> DavidsonResult:
    """Matrix-free symmetric Davidson (CPU, NumPy).

    This is a small, self-contained replacement for `pyscf.lib.davidson1` for
    symmetric/Hermitian problems, tailored to `cuguga` where `aop` naturally
    supports a batched list-of-vectors API.
    """

    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be >= 1")
    max_cycle = int(max_cycle)
    max_space = int(max_space)
    tol = float(tol)
    lindep = float(lindep)
    tol_residual = None if tol_residual is None else float(tol_residual)
    if max_cycle < 1:
        raise ValueError("max_cycle must be >= 1")
    if max_space < nroots:
        raise ValueError("max_space must be >= nroots")

    # Match the GPU Davidson behavior: allow extra room for multi-root expansions.
    max_space_eff = int(max_space + (nroots - 1) * 4)
    if max_space_eff < nroots:
        raise ValueError("invalid effective max_space")

    if not x0:
        raise ValueError("x0 must not be empty")
    x0_vec0 = _as_f64_vec(x0[0])
    n = int(x0_vec0.size)
    if n <= 0:
        raise ValueError("invalid vector size")

    toloose = float(np.sqrt(tol)) if tol_residual is None else float(tol_residual)

    # Basis matrices in column-major order so column views are contiguous.
    v = np.zeros((n, max_space_eff), dtype=np.float64, order="F")
    w = np.zeros((n, max_space_eff), dtype=np.float64, order="F")
    m = 0

    stats: dict[str, float] | None = {} if bool(profile) else None
    hop_calls = 0
    hop_time_s = 0.0
    orth_time_s = 0.0
    t_total0 = time.perf_counter() if stats is not None else 0.0

    def _aop_block_profiled(vmat: np.ndarray) -> np.ndarray:
        nonlocal hop_calls
        nonlocal hop_time_s
        if stats is None:
            return _aop_block(aop, vmat)

        if vmat.ndim != 2:
            raise ValueError("V must be 2D")
        n_loc, m_loc = map(int, vmat.shape)
        if n_loc <= 0 or m_loc <= 0:
            raise ValueError("invalid V shape")

        t0 = time.perf_counter()
        xs = [np.asarray(vmat[:, i], dtype=np.float64, order="C") for i in range(m_loc)]
        ys = aop(xs)
        if not isinstance(ys, list) or len(ys) != m_loc:
            raise ValueError("aop returned an unexpected result (expected list of length m)")
        out = np.empty((n_loc, m_loc), dtype=np.float64, order="F")
        for i, y in enumerate(ys):
            out[:, i] = _as_f64_vec(y, n=n_loc)
        hop_calls += int(m_loc)
        hop_time_s += time.perf_counter() - t0
        return out

    def _orthonormalize_one_profiled(
        vec: np.ndarray, *, basis: np.ndarray | None, lindep: float
    ) -> np.ndarray | None:
        nonlocal orth_time_s
        if stats is None:
            return _orthonormalize_one(vec, basis=basis, lindep=lindep)
        t0 = time.perf_counter()
        out = _orthonormalize_one(vec, basis=basis, lindep=lindep)
        orth_time_s += time.perf_counter() - t0
        return out

    for guess in x0:
        if m >= max_space_eff:
            break
        vg = _as_f64_vec(guess, n=n)
        vg = _orthonormalize_one_profiled(vg, basis=None if m == 0 else v[:, :m], lindep=lindep)
        if vg is None:
            continue
        v[:, m] = vg
        m += 1
        if m >= nroots:
            # For typical `cuguga` usage, `x0` already provides >= nroots guesses.
            # Avoid spending time orthonormalizing extra guesses unless needed.
            break

    if m < nroots:
        raise RuntimeError("failed to build enough linearly independent initial vectors")

    w[:, :m] = _aop_block_profiled(v[:, :m])

    conv = np.zeros((nroots,), dtype=np.bool_)
    e = np.zeros((nroots,), dtype=np.float64)
    e_prev: np.ndarray | None = None
    x = None
    niter = 0

    for _it in range(1, max_cycle + 1):
        niter += 1
        hsub = v[:, :m].T @ w[:, :m]
        hsub = 0.5 * (hsub + hsub.T)
        evals, u = np.linalg.eigh(hsub)
        e = np.asarray(evals[:nroots], dtype=np.float64)
        u_r = np.asarray(u[:, :nroots], dtype=np.float64)

        x = np.asarray(v[:, :m] @ u_r, dtype=np.float64, order="F")  # Ritz vectors
        ax = np.asarray(w[:, :m] @ u_r, dtype=np.float64, order="F")
        r = ax - x * e[None, :]
        rnorm = np.linalg.norm(r, axis=0)

        if e_prev is None:
            de = np.zeros_like(e)
        else:
            de = e - e_prev
        e_prev = e.copy()

        conv = (rnorm <= tol) | ((np.abs(de) <= tol) & (rnorm <= toloose))
        if bool(np.all(conv)):
            break

        new_vecs: list[np.ndarray] = []
        for root in range(nroots):
            if bool(conv[root]):
                continue
            t = precond(np.asarray(r[:, root], dtype=np.float64), float(e[root]), np.asarray(x[:, root], dtype=np.float64))
            t = _as_f64_vec(t, n=n)
            t = _orthonormalize_one_profiled(t, basis=v[:, :m], lindep=lindep)
            if t is None:
                continue
            if new_vecs:
                q = np.column_stack(new_vecs)
                t = _orthonormalize_one_profiled(t, basis=q, lindep=lindep)
                if t is None:
                    continue
            new_vecs.append(t)

        if not new_vecs:
            break

        if m + len(new_vecs) > max_space_eff:
            # Restart with the current Ritz vectors.
            v[:, :nroots] = x
            m = nroots
            w[:, :m] = _aop_block_profiled(v[:, :m])
            continue

        m0 = m
        for t in new_vecs:
            v[:, m] = t
            m += 1
        w[:, m0:m] = _aop_block_profiled(v[:, m0:m])

    if x is None:  # pragma: no cover
        raise RuntimeError("internal error: missing Ritz vectors")

    xs = [np.ascontiguousarray(x[:, i], dtype=np.float64) for i in range(int(nroots))]
    if stats is not None:
        total_time_s = time.perf_counter() - t_total0
        subspace_time_s = max(0.0, float(total_time_s) - float(hop_time_s) - float(orth_time_s))
        stats.update(
            {
                "hop_calls": float(hop_calls),
                "hop_time_s": float(hop_time_s),
                "orth_time_s": float(orth_time_s),
                "subspace_time_s": float(subspace_time_s),
            }
        )
    return DavidsonResult(
        converged=np.asarray(conv, dtype=np.bool_),
        e=np.asarray(e, dtype=np.float64),
        x=xs,
        niter=int(niter),
        stats=stats,
    )


def davidson1(
    aop: AOP,
    x0: list[Vec],
    precond: Precond,
    *,
    tol: float = 1e-10,
    lindep: float = 1e-14,
    max_cycle: int = 50,
    max_space: int = 12,
    nroots: int = 1,
    tol_residual: float | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Iterative Davidson diagonalization for symmetric matrices.

    Drop-in replacement for ``pyscf.lib.davidson1`` using a batched
    list-of-vectors ``aop`` interface.

    Parameters
    ----------
    aop : callable
        Matrix-vector product ``aop(xs) -> ys`` where *xs* and *ys* are
        lists of 1-D float64 arrays.
    x0 : list of np.ndarray
        Initial guess vectors (at least ``nroots`` linearly independent).
    precond : callable
        Preconditioner ``precond(residual, eigenvalue, eigenvector) -> vec``.
    tol : float, optional
        Convergence threshold on the eigenvalue change.
    lindep : float, optional
        Linear-dependence threshold for basis orthonormalization.
    max_cycle : int, optional
        Maximum number of Davidson iterations.
    max_space : int, optional
        Maximum subspace dimension before restart.
    nroots : int, optional
        Number of lowest eigenvalues/eigenvectors to compute.
    tol_residual : float or None, optional
        Explicit residual norm threshold. If *None*, defaults to
        ``sqrt(tol)``.

    Returns
    -------
    converged : np.ndarray
        Boolean array of shape ``(nroots,)``.
    e : np.ndarray
        Eigenvalues of shape ``(nroots,)``.
    x : list of np.ndarray
        Eigenvectors, each of length *n*.
    """
    res = _davidson_impl(
        aop,
        x0,
        precond,
        tol=float(tol),
        lindep=float(lindep),
        max_cycle=int(max_cycle),
        max_space=int(max_space),
        nroots=int(nroots),
        tol_residual=tol_residual,
        profile=False,
        **kwargs,
    )
    return res.converged, res.e, res.x


def davidson1_result(
    aop: AOP,
    x0: list[Vec],
    precond: Precond,
    *,
    tol: float = 1e-10,
    lindep: float = 1e-14,
    max_cycle: int = 50,
    max_space: int = 12,
    nroots: int = 1,
    tol_residual: float | None = None,
    profile: bool = True,
    **kwargs: Any,
) -> DavidsonResult:
    """Same as :func:`davidson1` but returns a :class:`DavidsonResult`.

    Parameters
    ----------
    aop, x0, precond, tol, lindep, max_cycle, max_space, nroots, tol_residual
        See :func:`davidson1`.
    profile : bool, optional
        If *True* (default), collect timing statistics in
        ``DavidsonResult.stats``.

    Returns
    -------
    DavidsonResult
        Structured result with convergence flags, eigenvalues, eigenvectors,
        iteration count, and optional profiling statistics.
    """
    return _davidson_impl(
        aop,
        x0,
        precond,
        tol=float(tol),
        lindep=float(lindep),
        max_cycle=int(max_cycle),
        max_space=int(max_space),
        nroots=int(nroots),
        tol_residual=tol_residual,
        profile=bool(profile),
        **kwargs,
    )
