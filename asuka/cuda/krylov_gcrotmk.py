from __future__ import annotations

"""Backend-agnostic GCROT(m,k) implementation for NumPy/CuPy arrays.

This module provides a small, self-contained implementation used by ASUKA's
Z-vector solver when a GPU-native Hessian matvec is available.
"""

from typing import Any, Callable, Literal
import math

import numpy as np

TruncateMode = Literal["oldest", "smallest"]


def _is_cupy_array(x: Any) -> bool:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception:
        return False
    return isinstance(x, cp.ndarray)


def _pick_xp(*objs: Any):
    for obj in objs:
        if obj is None:
            continue
        if _is_cupy_array(obj):
            import cupy as cp  # type: ignore[import-not-found]

            return cp
    return np


def _as_float(x: Any) -> float:
    try:
        return float(x.item())
    except Exception:
        return float(x)


def _norm(xp, x: Any) -> float:
    return _as_float(xp.linalg.norm(x))


def _vdot(xp, x: Any, y: Any) -> float:
    return _as_float(xp.vdot(x, y))


def _fgmres_givens_xp(
    matvec: Callable[[Any], Any],
    v0: Any,
    m: int,
    atol: float,
    *,
    rpsolve: Callable[[Any], Any] | None = None,
    cs: tuple[Any, ...] = (),
    xp=np,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Any], list[Any], np.ndarray, float]:
    """Flexible GMRES inner iteration with optional recycle-space projection."""

    if rpsolve is None:
        rpsolve = lambda x: x  # noqa: E731

    dtype_small = np.float64
    eps = np.finfo(np.float64).eps

    vs: list[Any] = [v0]
    zs: list[Any] = []
    n_c = len(cs)

    bmat = np.zeros((n_c, m), dtype=dtype_small)
    h_full = np.zeros((m + 1, m), dtype=dtype_small)
    h_rot = np.zeros((m + 1, m), dtype=dtype_small)

    giv_c = np.zeros(m, dtype=dtype_small)
    giv_s = np.zeros(m, dtype=dtype_small)
    g = np.zeros(m + 1, dtype=dtype_small)
    g[0] = 1.0

    j_last = -1
    res = float("nan")

    for j in range(m):
        z = rpsolve(vs[-1])
        w = matvec(z).copy()
        w_norm0 = _norm(xp, w)

        # Project against recycle C-space.
        for i, c in enumerate(cs):
            alpha = _vdot(xp, c, w)
            bmat[i, j] = alpha
            w -= alpha * c

        # Arnoldi vs V.
        for i, v in enumerate(vs):
            alpha = _vdot(xp, v, w)
            h_full[i, j] = alpha
            w -= alpha * v

        h_next = _norm(xp, w)
        h_full[j + 1, j] = h_next
        breakdown = not (h_next > eps * max(w_norm0, eps))

        if h_next != 0.0 and np.isfinite(h_next):
            v_next = w / h_next
        else:
            v_next = w

        vs.append(v_next)
        zs.append(z)

        # Apply prior Givens rotations to current column.
        h = h_full[: j + 2, j].copy()
        for i in range(j):
            c = giv_c[i]
            s = giv_s[i]
            hi = h[i]
            hip1 = h[i + 1]
            h[i] = c * hi + s * hip1
            h[i + 1] = -s * hi + c * hip1

        # Build new rotation zeroing h[j+1].
        a = float(h[j])
        b = float(h[j + 1])
        if b == 0.0:
            c_new = 1.0
            s_new = 0.0
        else:
            r = math.hypot(abs(a), abs(b))
            c_new = a / r
            s_new = b / r
        giv_c[j] = c_new
        giv_s[j] = s_new

        h[j] = c_new * a + s_new * b
        h[j + 1] = 0.0
        h_rot[: j + 2, j] = h

        gj = g[j]
        gj1 = g[j + 1]
        g[j] = c_new * gj + s_new * gj1
        g[j + 1] = -s_new * gj + c_new * gj1
        res = abs(float(g[j + 1]))
        j_last = j

        if res < atol or breakdown:
            break

    if j_last < 0:
        return (
            h_full[:, :0],
            h_rot[:, :0],
            bmat[:, :0],
            vs,
            zs,
            np.zeros((0,), dtype=dtype_small),
            float(res),
        )

    p = j_last + 1
    rtri = h_rot[:p, :p]
    rhs = g[:p]
    try:
        y = np.linalg.solve(rtri, rhs)
    except np.linalg.LinAlgError:
        y, *_ = np.linalg.lstsq(rtri, rhs, rcond=None)
    return h_full[:, :p], h_rot[:, :p], bmat[:, :p], vs, zs, y, float(res)


def _reorthonormalize_cu_inplace(
    cu: list[tuple[Any | None, Any]],
    *,
    matvec: Callable[[Any], Any],
    xp=np,
    drop_tol: float = 1e-12,
) -> None:
    """Rebuild C/U as an orthonormalized basis, preserving order."""

    new_cu: list[tuple[Any, Any]] = []
    first_norm: float | None = None

    for c, u in cu:
        if c is None:
            c = matvec(u)
        c = c.copy()
        u = u.copy()

        for cp_vec, up_vec in new_cu:
            alpha = _vdot(xp, cp_vec, c)
            c -= alpha * cp_vec
            u -= alpha * up_vec

        cn = _norm(xp, c)
        if first_norm is None:
            first_norm = cn

        if first_norm is None or first_norm == 0.0:
            continue
        if cn < drop_tol * first_norm:
            continue

        c /= cn
        u /= cn
        new_cu.append((c, u))

    cu[:] = new_cu


def gcrotmk_xp(
    matvec: Callable[[Any], Any],
    b: Any,
    x0: Any | None = None,
    *,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int = 1000,
    M: Callable[[Any], Any] | None = None,
    callback: Callable[[Any], None] | None = None,
    m: int = 20,
    k: int | None = None,
    CU: list[tuple[Any | None, Any]] | None = None,
    discard_C: bool = False,
    truncate: TruncateMode = "oldest",
    xp: Any = None,
) -> tuple[Any, int]:
    """GCROT(m,k) solve for NumPy/CuPy vectors.

    Returns
    -------
    x : Any
        Solution vector in the same backend as `b`.
    info : int
        0 on convergence; positive outer-iteration count on non-convergence.
    """

    if CU is None:
        CU = []
    if xp is None:
        xp = _pick_xp(b, x0, *(c for c, _ in CU), *(u for _, u in CU))
    if M is None:
        M = lambda x: x  # noqa: E731

    b = xp.asarray(b, dtype=xp.float64).ravel()
    if x0 is None:
        x = xp.zeros_like(b)
    else:
        x = xp.asarray(x0, dtype=xp.float64).ravel().copy()

    b_norm = _norm(xp, b)
    if b_norm == 0.0:
        return b, 0

    if discard_C:
        CU[:] = [(None, u) for (c, u) in CU]

    r = b - matvec(x)

    if CU:
        _reorthonormalize_cu_inplace(CU, matvec=matvec, xp=xp)
        for c, u in CU:
            yc = _vdot(xp, c, r)
            x += u * yc
            r -= c * yc

    if k is None:
        k = int(m)
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")

    def _append_solution_to_cu() -> None:
        while len(CU) >= k and CU:
            del CU[0]
        CU.append((None, x.copy()))

    beta_tol_const = max(float(atol), float(rtol) * float(b_norm))
    maxiter_i = int(maxiter)
    m_i = int(m)
    if m_i <= 0:
        raise ValueError("m must be positive")

    for j_outer in range(maxiter_i):
        if callback is not None:
            callback(x)

        beta = _norm(xp, r)
        beta_tol = beta_tol_const

        if beta <= beta_tol and (j_outer > 0 or CU):
            r = b - matvec(x)
            beta = _norm(xp, r)

        if beta <= beta_tol:
            _append_solution_to_cu()
            if discard_C:
                CU[:] = [(None, u) for (c, u) in CU]
            return x, 0

        ml = m_i + max(k - len(CU), 0)
        cs = tuple(c for (c, _u) in CU)

        v0 = r / beta
        h_full, h_rot, bmat, vs, zs, y, _pres = _fgmres_givens_xp(
            matvec,
            v0,
            ml,
            beta_tol / beta,
            rpsolve=M,
            cs=cs,
            xp=xp,
        )

        y = y * float(beta)
        p = int(y.size)
        if p == 0:
            _append_solution_to_cu()
            if discard_C:
                CU[:] = [(None, u) for (c, u) in CU]
            return x, j_outer + 1

        ux = zs[0] * y[0]
        for z_i, yi in zip(zs[1:], y[1:]):
            ux += z_i * yi

        if CU:
            by = bmat.dot(y)
            for (c, u), byc in zip(CU, by):
                ux -= u * float(byc)

        hy = h_full.dot(y)
        cx = vs[0] * float(hy[0])
        for v_i, hyi in zip(vs[1:], hy[1:]):
            cx += v_i * float(hyi)

        cx_norm = _norm(xp, cx)
        if cx_norm == 0.0 or not np.isfinite(cx_norm):
            continue

        alpha = 1.0 / float(cx_norm)
        cx *= alpha
        ux *= alpha

        gamma = _vdot(xp, cx, r)
        r -= cx * gamma
        x += ux * gamma

        if truncate == "oldest":
            while len(CU) >= k and CU:
                del CU[0]
        elif truncate == "smallest":
            if len(CU) >= k and k > 1 and p > 1:
                rtri = h_rot[:p, :p]
                try:
                    dmat = np.linalg.solve(rtri[:-1, :].T, bmat.T).T
                except np.linalg.LinAlgError:
                    dmat = np.linalg.lstsq(rtri[:-1, :].T, bmat.T, rcond=None)[0].T
                wmat, _sigma, _vh = np.linalg.svd(dmat, full_matrices=False)

                new_cu: list[tuple[Any, Any]] = []
                for w in wmat[:, : k - 1].T:
                    c0, u0 = CU[0]
                    c = c0 * float(w[0])
                    u = u0 * float(w[0])
                    for (cp_vec, up_vec), wp in zip(CU[1:], w[1:]):
                        c += cp_vec * float(wp)
                        u += up_vec * float(wp)
                    for c_prev, u_prev in new_cu:
                        a = _vdot(xp, c_prev, c)
                        c -= c_prev * a
                        u -= u_prev * a
                    cn = _norm(xp, c)
                    if cn == 0.0:
                        continue
                    c /= cn
                    u /= cn
                    new_cu.append((c, u))
                CU[:] = new_cu
        else:
            raise ValueError("truncate must be 'oldest' or 'smallest'")

        CU.append((cx, ux))

    _append_solution_to_cu()
    if discard_C:
        CU[:] = [(None, u) for (c, u) in CU]
    return x, maxiter_i


__all__ = ["gcrotmk_xp"]
