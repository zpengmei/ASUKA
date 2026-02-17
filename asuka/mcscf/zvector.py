from __future__ import annotations

"""Z-vector (Lagrangian) utilities for CASSCF objects.

This module provides a *thin but practical* implementation of the Z-vector
formalism for MCSCF/CASSCF wavefunctions.

Why this exists
---------------
`cuguga` accelerates the active-space solver (GUGA/CSF) and DF integral backend.
For analytic gradients of post-CASSCF methods (CASPT2/NEVPT2/...) you typically
need a Z-vector solve in the *reference* CASSCF parameter space (orbitals + CI).

PySCF already has robust code for building the CASSCF super-Hessian operator
(`mc.gen_g_hop`).  This module focuses on:

1) Turning that operator into a linear system solve (iterative GMRES).
2) Providing helper routines to form the CI-part right-hand side from an
   "effective Hamiltonian" (i.e. derivatives with respect to active-space RDMs).
3) Providing helper routines to convert the CI Z-vector into effective/relaxed
   active-space RDM contributions via transition RDMs.

Design notes
------------
* **Lazy PySCF import**: PySCF is imported inside functions only.
* **No assumptions about determinantal vs CSF solvers**: we only require the
  PySCF FCI-solver interface methods that `asuka.solver.GUGAFCISolver` already
  implements (`contract_2e`, `trans_rdm12`, ...).
* **State-average support** is non-trivial (orthonormal constraints between
  multiple roots).  The helper routines here are primarily aimed at the
  *single-state/state-specific* case.  The linear solver wrapper itself is
  general as long as `mc.gen_g_hop` returns a compatible Hessian operator.
"""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import time


@dataclass(frozen=True)
class MCSCFZVectorResult:
    """Container for a solved MCSCF/CASSCF Z-vector.

    Attributes
    ----------
    converged : bool
        Whether the solver converged.
    niter : int
        Number of iterations.
    residual_norm : float
        Final residual norm.
    z_orb : np.ndarray
        Orbital Z-vector (packed).
    z_ci : Any
        CI Z-vector (structure matches input CI).
    z_packed : np.ndarray
        The raw packed vector as used in the linear solve.
    info : dict[str, Any]
        Solver backend info (e.g. GMRES return code).
    """

    converged: bool
    niter: int
    residual_norm: float

    z_orb: np.ndarray
    z_ci: Any

    z_packed: np.ndarray

    info: dict[str, Any]


@dataclass(frozen=True)
class MCSCFHessianOp:
    """Linear operator representation of the MCSCF/CASSCF super-Hessian.

    Attributes
    ----------
    mv : Callable[[np.ndarray], np.ndarray]
        Matrix-vector product function.
    diag : np.ndarray | None
        Diagonal preconditioner (approximate).
    n_orb : int
        Number of orbital rotation variables.
    n_ci : int
        Number of CI variables.
    ci_template : Any
        Template CI object for structure.
    ci_unflatten : Callable[[np.ndarray], Any]
        Function to unflatten CI vector.
    orb_only : bool
        Whether the operator acts only on orbitals.
    is_sa : bool
        Whether this is for state-averaged CASSCF.
    ci_ref_list : list[np.ndarray] | None
        Reference CI vectors for SA projection.
    sa_gram_inv : np.ndarray | None
        Inverse Gram matrix for SA projection.

    Notes
    -----
    The operator acts on *packed* orbital-rotation variables plus (optionally)
    packed CI variables. For some `mc` objects where only the orbital Hessian
    action is available (via ``mc.gen_g_hop``), `orb_only=True` and the operator
    acts only on the orbital variables.
    """

    mv: Callable[[np.ndarray], np.ndarray]
    diag: np.ndarray | None

    n_orb: int
    n_ci: int
    ci_template: Any
    ci_unflatten: Callable[[np.ndarray], Any]

    orb_only: bool
    is_sa: bool
    ci_ref_list: list[np.ndarray] | None
    sa_gram_inv: np.ndarray | None = None
    gpu_mode: bool = False  # True when h_op supports CuPy arrays natively

    @property
    def n_tot(self) -> int:
        return int(self.n_orb + self.n_ci)


@contextmanager
def _maybe_set_attr(obj: Any, name: str, value: Any):
    """Best-effort temporary attribute assignment (restores original value)."""

    missing = object()
    changed = False
    old = missing
    if obj is None:
        yield False
        return

    try:
        old = getattr(obj, name, missing)
        if old is missing or old != value:
            setattr(obj, name, value)
            changed = True
    except Exception:
        yield False
        return

    try:
        yield changed
    finally:
        if not changed:
            return
        try:
            if old is missing:
                delattr(obj, name)
            else:
                setattr(obj, name, old)
        except Exception:
            pass


def _flatten_ci(ci: Any) -> tuple[np.ndarray, Callable[[np.ndarray], Any]]:
    """Flatten CI-like objects.

    Parameters
    ----------
    ci : Any
        Either a 1D ndarray (single root) or a list/tuple of 1D ndarrays
        (multi-root).

    Returns
    -------
    vec : np.ndarray
        1D float64 vector.
    unflatten : Callable[[np.ndarray], Any]
        Function that maps a 1D vector back to the original structure.
    """

    if isinstance(ci, np.ndarray):
        arr = np.asarray(ci, dtype=np.float64)
        shape = arr.shape
        if arr.ndim == 1:
            def _unflatten(v: np.ndarray) -> np.ndarray:
                vv = np.asarray(v, dtype=np.float64).ravel()
                if vv.size != arr.size:
                    raise ValueError("CI size mismatch")
                return vv.reshape(shape)

            return arr.ravel(), _unflatten
        if arr.ndim == 2:
            # Accept PySCF's occasional (1,ncsf) wrappers.
            def _unflatten(v: np.ndarray) -> np.ndarray:
                vv = np.asarray(v, dtype=np.float64).ravel()
                if vv.size != arr.size:
                    raise ValueError("CI size mismatch")
                return vv.reshape(shape)

            return arr.ravel(), _unflatten
        raise ValueError("Unsupported CI ndarray ndim")

    if isinstance(ci, (list, tuple)):
        vecs = [np.asarray(v, dtype=np.float64).ravel() for v in ci]
        sizes = [int(v.size) for v in vecs]
        total = int(sum(sizes))
        if total == 0:
            raise ValueError("Empty CI list")

        flat = np.concatenate(vecs)

        def _unflatten(v: np.ndarray) -> list[np.ndarray]:
            vv = np.asarray(v, dtype=np.float64).ravel()
            if vv.size != total:
                raise ValueError("CI size mismatch")
            out: list[np.ndarray] = []
            off = 0
            for s in sizes:
                out.append(vv[off:off + s].copy())
                off += s
            return out

        return flat, _unflatten

    raise ValueError(f"Unsupported CI type: {type(ci)!r}")


def ensure_real_ci_rhs(rhs_ci: Any, *, imag_tol: float = 1e-10) -> Any:
    """Convert a CI RHS to float64, erroring out if the imaginary part is too large.

    This is useful for SOC/SI workflows where intermediates are complex, but the
    final variational response RHS entering the (real) MCSCF Z-vector equation is
    expected to be real.

    Parameters
    ----------
    rhs_ci : Any
        CI RHS vector(s).
    imag_tol : float, optional
        Tolerance for imaginary component magnitude.

    Returns
    -------
    Any
        Real part of CI RHS.
    """

    if isinstance(rhs_ci, (list, tuple)):
        return [ensure_real_ci_rhs(v, imag_tol=float(imag_tol)) for v in rhs_ci]

    arr = np.asarray(rhs_ci)
    if np.iscomplexobj(arr):
        max_imag = float(np.max(np.abs(arr.imag))) if arr.size else 0.0
        if max_imag > float(imag_tol):
            raise ValueError(f"rhs_ci has large imaginary part: max|Im|={max_imag:g} > {imag_tol:g}")
        arr = arr.real
    return np.asarray(arr, dtype=np.float64)


def project_ci_rhs_normalized(ci0: Any, rhs_ci: Any) -> Any:
    """Project CI RHS vectors to the tangent space of normalized CI vectors.

    For a real, normalized CI vector ``c`` with constraint ``c^T c = 1``, the
    physically meaningful CI gradient/RHS lives in the orthogonal complement of
    ``c``. This helper removes the component parallel to ``ci0``:

        rhs <- rhs - (ci0·rhs)/(ci0·ci0) * ci0

    Multi-root inputs (lists/tuples) are handled by projecting each RHS vector to the
    orthogonal complement of the *entire* root subspace spanned by ``ci0``.  This is
    important for state-average CASSCF, where CI variations are constrained by
    inter-root orthonormality and where the SA response equations are singular in the
    root-subspace directions.

    Parameters
    ----------
    ci0 : Any
        Reference CI vector(s).
    rhs_ci : Any
        RHS CI vector(s).

    Returns
    -------
    Any
        Projected RHS CI vector(s).
    """

    if isinstance(ci0, (list, tuple)):
        if not isinstance(rhs_ci, (list, tuple)) or len(rhs_ci) != len(ci0):
            raise ValueError("rhs_ci must have the same list/tuple structure as ci0")

        c_list = [np.asarray(c, dtype=np.float64).ravel() for c in ci0]
        sizes = {int(c.size) for c in c_list}
        if len(sizes) != 1:
            # Fallback: inconsistent CI sizes (should not happen in SA-CASSCF); project root-by-root.
            return [project_ci_rhs_normalized(c, g) for c, g in zip(ci0, rhs_ci)]

        # Project against the span of all CI roots: g <- g - C (C^T C)^(-1) C^T g
        cmat = np.stack(c_list, axis=1)  # (nci, nroots)
        gram = cmat.T @ cmat
        try:
            gram_inv = np.linalg.inv(gram)
        except np.linalg.LinAlgError:  # pragma: no cover
            gram_inv = np.linalg.pinv(gram)

        g_list = [np.asarray(g, dtype=np.float64) for g in rhs_ci]
        shapes = [g.shape for g in g_list]
        gmat = np.stack([g.ravel() for g in g_list], axis=1)  # (nci, nvec)
        if int(gmat.shape[0]) != int(cmat.shape[0]):
            raise ValueError("CI RHS size mismatch")
        coeff = cmat.T @ gmat  # (nroots, nvec)
        gmat = gmat - cmat @ (gram_inv @ coeff)
        return [np.ascontiguousarray(gmat[:, i].reshape(shapes[i])) for i in range(int(gmat.shape[1]))]

    c0 = np.asarray(ci0, dtype=np.float64).ravel()
    g0 = np.asarray(rhs_ci, dtype=np.float64)
    shape = g0.shape
    g = g0.ravel().copy()
    if c0.size != g.size:
        raise ValueError("CI RHS size mismatch")
    denom = float(np.dot(c0, c0))
    if denom > 0.0:
        alpha = float(np.dot(c0, g)) / denom
        g -= alpha * c0
    return g.reshape(shape)


def prepare_ci_rhs_for_zvector(
    *,
    ci0: Any,
    rhs_ci: Any,
    imag_tol: float = 1e-10,
    project_normalized: bool = True,
) -> Any:
    """Prepare a CI RHS for `solve_mcscf_zvector` (real dtype, optional projection).

    Parameters
    ----------
    ci0 : Any
        Reference CI vector.
    rhs_ci : Any
        RHS CI vector.
    imag_tol : float, optional
        Tolerance for imaginary parts.
    project_normalized : bool, optional
        If True, project RHS to tangent space.

    Returns
    -------
    Any
        Prepared RHS.
    """

    rhs = ensure_real_ci_rhs(rhs_ci, imag_tol=float(imag_tol))
    if project_normalized:
        rhs = project_ci_rhs_normalized(ci0, rhs)
    return rhs


def _as_rhs_or_zeros(rhs: Any | None, template: Any) -> tuple[np.ndarray, Callable[[np.ndarray], Any]]:
    """Return a flattened RHS vector, defaulting to zeros with template shape."""

    t_flat, t_unflatten = _flatten_ci(template)
    if rhs is None:
        return np.zeros_like(t_flat), t_unflatten
    r_flat, r_unflatten = _flatten_ci(rhs)
    if r_flat.size != t_flat.size:
        raise ValueError("rhs_ci shape mismatch")
    # Use the template unflatten, to preserve the template's exact structure.
    return r_flat, t_unflatten


def _wrap_h_op(
    h_op: Any,
    n_orb: int,
    ci_template: Any,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap PySCF's `h_op` to a function that takes/returns flat vectors."""

    ci0_flat, ci_unflatten = _flatten_ci(ci_template)
    n_ci = int(ci0_flat.size)
    n_tot = int(n_orb + n_ci)

    def _flatten_out(y: Any) -> np.ndarray:
        if isinstance(y, np.ndarray):
            yy = np.asarray(y, dtype=np.float64).ravel()
            if yy.size == n_tot:
                return yy
            raise ValueError("Unexpected h_op output size")
        if isinstance(y, (list, tuple)):
            if len(y) != 2:
                raise ValueError("Expected h_op output (orb, ci)")
            y_orb = np.asarray(y[0], dtype=np.float64).ravel()
            y_ci, _ = _flatten_ci(y[1])
            if y_orb.size != n_orb or y_ci.size != n_ci:
                raise ValueError("Unexpected h_op output shapes")
            return np.concatenate([y_orb, y_ci])
        raise ValueError(f"Unsupported h_op output type: {type(y)!r}")

    def _call(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != n_tot:
            raise ValueError("h_op input length mismatch")

        # Try the common PySCF convention: h_op takes a single packed vector.
        try:
            y = h_op(x)
        except TypeError:
            # Fallback: some implementations may expect (x_orb, x_ci)
            x_orb = x[:n_orb]
            x_ci = ci_unflatten(x[n_orb:])
            y = h_op(x_orb, x_ci)

        return _flatten_out(y)

    return _call


def _project_sa_ci_components(
    ci0: list[np.ndarray],
    vecs: list[np.ndarray],
    *,
    gram_inv: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Project CI-like vectors to the orthogonal complement of the SA root subspace.

    This mirrors PySCF's SA-CASSCF Lagrange machinery (`sacasscf.Gradients.project_Aop`),
    which removes components parallel to the state-average CI span. The SA super-Hessian
    is singular in these directions, and the Z-vector is not uniquely defined unless a
    gauge is fixed.

    Notes
    -----
    This routine assumes all CI roots live in the same CI space (same flattened size).
    Supports both numpy and CuPy arrays — dispatches based on input type.
    """

    # Detect array backend from input vectors.
    try:
        import cupy as _cp_proj  # type: ignore[import-not-found]
    except Exception:
        _cp_proj = None
    _on_gpu = False
    if _cp_proj is not None:
        for v in vecs:
            if isinstance(v, _cp_proj.ndarray):
                _on_gpu = True
                break
        if not _on_gpu:
            for c in ci0:
                if isinstance(c, _cp_proj.ndarray):
                    _on_gpu = True
                    break
    xp = _cp_proj if _on_gpu else np

    c_list = [xp.asarray(c, dtype=xp.float64).ravel() for c in ci0]
    sizes = {int(c.size) for c in c_list}
    if len(sizes) != 1:
        return vecs
    nci = int(next(iter(sizes)))
    nroots = int(len(c_list))
    if nroots <= 0:
        return vecs

    if gram_inv is None:
        gram = xp.empty((nroots, nroots), dtype=xp.float64)
        for i in range(nroots):
            ci = c_list[i]
            gram[i, i] = xp.dot(ci, ci)
            for j in range(i + 1, nroots):
                cij = xp.dot(ci, c_list[j])
                gram[i, j] = cij
                gram[j, i] = cij
        try:
            gram_inv_use = xp.linalg.inv(gram)
        except xp.linalg.LinAlgError:  # pragma: no cover
            gram_inv_use = xp.linalg.pinv(gram)
    else:
        gram_inv_use = xp.asarray(gram_inv, dtype=xp.float64)
        if gram_inv_use.shape != (nroots, nroots):
            raise ValueError("SA gram_inv shape mismatch in projection")

    out: list = []
    for v in vecs:
        v_arr = xp.asarray(v, dtype=xp.float64)
        shape = v_arr.shape
        v_flat = v_arr.ravel()
        if int(v_flat.size) != nci:
            raise ValueError("SA CI vector size mismatch in projection")

        coeff = xp.empty(nroots, dtype=xp.float64)
        for i in range(nroots):
            coeff[i] = xp.dot(c_list[i], v_flat)
        alpha = gram_inv_use @ coeff

        v_proj = v_flat.copy()
        for i in range(nroots):
            v_proj -= alpha[i] * c_list[i]
        out.append(xp.ascontiguousarray(v_proj.reshape(shape)))

    return out


def _extract_h_diag(h_diag: Any, n_orb: int, ci_template: Any) -> np.ndarray | None:
    """Best-effort extraction of a diagonal preconditioner from PySCF outputs."""

    if h_diag is None:
        return None

    ci_flat, _ = _flatten_ci(ci_template)
    n_ci = int(ci_flat.size)
    n_tot = int(n_orb + n_ci)

    if isinstance(h_diag, np.ndarray):
        d = np.asarray(h_diag, dtype=np.float64).ravel()
        if d.size == n_tot:
            return d
        # Some PySCF paths return only the orbital diagonal.
        if d.size == n_orb:
            out = np.ones(n_tot, dtype=np.float64)
            out[:n_orb] = d
            return out
        raise ValueError("Unexpected h_diag size")

    if isinstance(h_diag, (list, tuple)):
        # Common: (diag_orb, diag_ci)
        if len(h_diag) == 2:
            d_orb = np.asarray(h_diag[0], dtype=np.float64).ravel()
            d_ci = np.asarray(_flatten_ci(h_diag[1])[0], dtype=np.float64).ravel()
            if d_orb.size != n_orb or d_ci.size != n_ci:
                raise ValueError("Unexpected h_diag tuple shapes")
            return np.concatenate([d_orb, d_ci])
        raise ValueError("Unsupported h_diag tuple")

    # Unknown type.
    return None


def _gmres_solve_gpu(
    mv: Callable,
    b: np.ndarray,
    *,
    diag_precond: np.ndarray | None = None,
    precond: Callable | None = None,
    tol: float = 1e-10,
    maxiter: int = 200,
    restart: int | None = None,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """GPU-native GMRES solve using CuPy — zero GPU↔CPU sync during iteration."""

    import cupy as cp  # type: ignore[import-not-found]
    from cupyx.scipy.sparse.linalg import LinearOperator as CuLinearOperator
    from cupyx.scipy.sparse.linalg import gmres as cu_gmres

    b_np = np.asarray(b, dtype=np.float64).ravel()
    n = int(b_np.size)
    bnorm = float(np.linalg.norm(b_np))
    if n == 0:
        return b_np.copy(), {
            "info": 0, "niter": 0, "matvec_calls": 0,
            "matvec_time_total": 0.0, "matvec_time_avg": 0.0,
            "solve_time_total": 0.0, "rhs_norm": 0.0,
            "residual_norm": 0.0, "residual_rel": 0.0,
            "solver": "gmres_gpu",
        }

    b_d = cp.asarray(b_np, dtype=cp.float64)

    matvec_calls = 0
    matvec_time = 0.0

    def _mv_gpu(x_d):
        nonlocal matvec_calls, matvec_time
        matvec_calls += 1
        t0 = time.perf_counter()
        y_d = mv(x_d)
        # Ensure result is CuPy float64.
        y_d = cp.asarray(y_d, dtype=cp.float64).ravel()
        matvec_time += time.perf_counter() - t0
        return y_d

    A = CuLinearOperator((n, n), matvec=_mv_gpu, dtype=cp.float64)

    # Preconditioner on GPU.
    M = None
    if precond is not None:
        # precond is CPU-based (e.g. SA Lagrange); roundtrip through numpy.
        def _m_mv_gpu(x_d):
            x_np = cp.asnumpy(x_d).astype(np.float64)
            y_np = np.asarray(precond(x_np), dtype=np.float64).ravel()
            return cp.asarray(y_np, dtype=cp.float64)
        M = CuLinearOperator((n, n), matvec=_m_mv_gpu, dtype=cp.float64)
    elif diag_precond is not None:
        d = cp.asarray(diag_precond, dtype=cp.float64).ravel()
        if d.size != n:
            raise ValueError("diag_precond length mismatch")
        d_safe = cp.where(cp.abs(d) > 1e-14, d, 1.0)
        def _m_mv_gpu(x_d):
            return x_d / d_safe
        M = CuLinearOperator((n, n), matvec=_m_mv_gpu, dtype=cp.float64)

    x0_d = None
    if x0 is not None:
        x0_d = cp.asarray(x0, dtype=cp.float64).ravel()
        if x0_d.size != n:
            raise ValueError("x0 length mismatch")

    restart_val = restart if restart is not None else min(n, 30)

    t_solve0 = time.perf_counter()
    # CuPy GMRES: all iterations on GPU, zero sync.
    x_d, info = cu_gmres(
        A, b_d,
        x0=x0_d,
        tol=float(tol),
        maxiter=int(maxiter),
        restart=int(restart_val),
        M=M,
    )
    solve_time = time.perf_counter() - t_solve0

    # Single sync at end of entire solve.
    x = cp.asnumpy(x_d)
    x = np.asarray(x, dtype=np.float64).ravel()

    # Compute residual on CPU.
    r_d = mv(cp.asarray(x, dtype=cp.float64)) - b_d
    resid = float(cp.linalg.norm(r_d))
    rel = resid / bnorm if bnorm > 0.0 else resid

    out: dict[str, Any] = {
        "info": int(info),
        "niter": int(matvec_calls),  # CuPy doesn't expose iteration count separately
        "matvec_calls": int(matvec_calls),
        "matvec_time_total": float(matvec_time),
        "matvec_time_avg": float(matvec_time) / float(matvec_calls) if matvec_calls else 0.0,
        "solve_time_total": float(solve_time),
        "rhs_norm": float(bnorm),
        "residual_norm": resid,
        "residual_rel": float(rel),
        "solver": "gmres_gpu",
    }
    return x, out


def _gmres_solve(
    mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    diag_precond: np.ndarray | None = None,
    precond: Callable[[np.ndarray], np.ndarray] | None = None,
    tol: float = 1e-10,
    maxiter: int = 200,
    restart: int | None = None,
    x0: np.ndarray | None = None,
    use_gpu: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve A x = b with restarted GMRES.

    Parameters
    ----------
    mv : Callable[[np.ndarray], np.ndarray]
        Matrix-vector product function.
    b : np.ndarray
        RHS vector.
    diag_precond : np.ndarray | None, optional
        Diagonal preconditioner array.
    precond : Callable[[np.ndarray], np.ndarray] | None, optional
        Preconditioner function.
    tol : float, optional
        Convergence tolerance.
    maxiter : int, optional
        Max iterations.
    restart : int | None, optional
        Restart size.
    x0 : np.ndarray | None, optional
        Initial guess.

    Returns
    -------
    x : np.ndarray
        Solution vector.
    info : dict[str, Any]
        Solver info.

    Notes
    -----
    * We use GMRES instead of CG/MINRES because the CASSCF super-Hessian is often
      indefinite and the effective operator returned by PySCF is not guaranteed
      to be SPD.
    """

    if use_gpu:
        return _gmres_solve_gpu(
            mv, b, diag_precond=diag_precond, precond=precond,
            tol=tol, maxiter=maxiter, restart=restart, x0=x0,
        )

    from scipy.sparse.linalg import LinearOperator, gmres

    b = np.asarray(b, dtype=np.float64).ravel()
    n = int(b.size)
    bnorm = float(np.linalg.norm(b))
    if n == 0:
        return b.copy(), {
            "info": 0,
            "niter": 0,
            "matvec_calls": 0,
            "matvec_time_total": 0.0,
            "matvec_time_avg": 0.0,
            "solve_time_total": 0.0,
            "rhs_norm": 0.0,
            "residual_norm": 0.0,
            "residual_rel": 0.0,
            "solver": "gmres",
        }

    matvec_calls = 0
    matvec_time = 0.0

    def _mv_counted(x: np.ndarray) -> np.ndarray:
        nonlocal matvec_calls
        nonlocal matvec_time
        matvec_calls += 1
        t0 = time.perf_counter()
        y = mv(x)
        matvec_time += time.perf_counter() - t0
        return y

    A = LinearOperator((n, n), matvec=_mv_counted, dtype=np.float64)

    M = None
    if precond is not None:
        def _m_mv(x: np.ndarray) -> np.ndarray:
            return np.asarray(precond(np.asarray(x, dtype=np.float64).ravel()), dtype=np.float64).ravel()

        M = LinearOperator((n, n), matvec=_m_mv, dtype=np.float64)
    elif diag_precond is not None:
        d = np.asarray(diag_precond, dtype=np.float64).ravel()
        if d.size != n:
            raise ValueError("diag_precond length mismatch")
        # Avoid division by 0.
        d_safe = np.where(np.abs(d) > 1e-14, d, 1.0)

        def _m_mv(x: np.ndarray) -> np.ndarray:
            return np.asarray(x, dtype=np.float64).ravel() / d_safe

        M = LinearOperator((n, n), matvec=_m_mv, dtype=np.float64)

    niter = 0

    def _cb(_rk=None):
        nonlocal niter
        niter += 1

    x0_use = None if x0 is None else np.asarray(x0, dtype=np.float64).ravel()
    if x0_use is not None and int(x0_use.size) != n:
        raise ValueError("x0 length mismatch")

    kwargs: dict[str, Any] = {
        "M": M,
        "rtol": float(tol),
        "atol": 0.0,
        "maxiter": int(maxiter),
        "callback": _cb,
    }
    if restart is not None:
        kwargs["restart"] = int(restart)
    if x0_use is not None:
        kwargs["x0"] = x0_use

    # SciPy's gmres (>=1.12) uses `rtol`/`atol`.
    # Prefer the modern `callback_type` if available, but keep compatibility with older SciPy.
    t_solve0 = time.perf_counter()
    try:
        x, info = gmres(A, b, callback_type="pr_norm", **kwargs)
    except TypeError:  # pragma: no cover
        # Older SciPy versions: no `callback_type`.
        kwargs.pop("callback_type", None)
        x, info = gmres(A, b, **kwargs)
    solve_time = time.perf_counter() - t_solve0
    x = np.asarray(x, dtype=np.float64).ravel()
    r = mv(x) - b
    resid = float(np.linalg.norm(r))
    rel = resid / bnorm if bnorm > 0.0 else resid

    # `info` semantics: 0 converged, >0 = iteration count at restart, <0 breakdown.
    out: dict[str, Any] = {
        "info": int(info),
        "niter": int(niter),
        "matvec_calls": int(matvec_calls),
        "matvec_time_total": float(matvec_time),
        "matvec_time_avg": float(matvec_time) / float(matvec_calls) if matvec_calls else 0.0,
        "solve_time_total": float(solve_time),
        "rhs_norm": float(bnorm),
        "residual_norm": resid,
        "residual_rel": float(rel),
        "solver": "gmres",
    }
    return x, out


def _gcrotmk_solve(
    mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    diag_precond: np.ndarray | None = None,
    precond: Callable[[np.ndarray], np.ndarray] | None = None,
    tol: float = 1e-10,
    maxiter: int = 50,
    m: int | None = None,
    k: int | None = None,
    x0: np.ndarray | None = None,
    recycle_space: list[tuple[np.ndarray | None, np.ndarray]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve A x = b with flexible GCROT(m,k), optionally recycling subspace vectors.

    Parameters
    ----------
    mv : Callable[[np.ndarray], np.ndarray]
        Matrix-vector product function.
    b : np.ndarray
        RHS vector.
    diag_precond : np.ndarray | None, optional
        Diagonal preconditioner.
    precond : Callable[[np.ndarray], np.ndarray] | None, optional
        Preconditioner function.
    tol : float, optional
        Convergence tolerance.
    maxiter : int, optional
        Max iterations.
    m : int | None, optional
        Number of inner GMRES iterations per cycle.
    k : int | None, optional
        Number of vectors to carry over from the previous cycle.
    x0 : np.ndarray | None, optional
        Initial guess.
    recycle_space : list | None, optional
        Recycling subspace (mutable list).

    Returns
    -------
    x : np.ndarray
        Solution vector.
    info : dict[str, Any]
        Solver info.

    Notes
    -----
    SciPy's ``gcrotmk`` supports reusing a recycle space across multiple solves
    through the mutable ``CU`` list. This is a key tool for multi-RHS workflows
    (all-roots gradients) where the operator is constant but the RHS changes.
    """

    from scipy.sparse.linalg import LinearOperator, gcrotmk

    b = np.asarray(b, dtype=np.float64).ravel()
    n = int(b.size)
    bnorm = float(np.linalg.norm(b))
    if n == 0:
        return b.copy(), {
            "info": 0,
            "niter": 0,
            "matvec_calls": 0,
            "matvec_time_total": 0.0,
            "matvec_time_avg": 0.0,
            "solve_time_total": 0.0,
            "rhs_norm": 0.0,
            "residual_norm": 0.0,
            "residual_rel": 0.0,
            "solver": "gcrotmk",
        }

    matvec_calls = 0
    matvec_time = 0.0

    def _mv_counted(x: np.ndarray) -> np.ndarray:
        nonlocal matvec_calls
        nonlocal matvec_time
        matvec_calls += 1
        t0 = time.perf_counter()
        y = mv(x)
        matvec_time += time.perf_counter() - t0
        return y

    A = LinearOperator((n, n), matvec=_mv_counted, dtype=np.float64)

    M = None
    if precond is not None:
        def _m_mv(x: np.ndarray) -> np.ndarray:
            return np.asarray(precond(np.asarray(x, dtype=np.float64).ravel()), dtype=np.float64).ravel()

        M = LinearOperator((n, n), matvec=_m_mv, dtype=np.float64)
    elif diag_precond is not None:
        d = np.asarray(diag_precond, dtype=np.float64).ravel()
        if d.size != n:
            raise ValueError("diag_precond length mismatch")
        d_safe = np.where(np.abs(d) > 1e-14, d, 1.0)

        def _m_mv(x: np.ndarray) -> np.ndarray:
            return np.asarray(x, dtype=np.float64).ravel() / d_safe

        M = LinearOperator((n, n), matvec=_m_mv, dtype=np.float64)

    niter = 0

    def _cb(_xk=None):
        nonlocal niter
        niter += 1

    x0_use = None if x0 is None else np.asarray(x0, dtype=np.float64).ravel()
    if x0_use is not None and int(x0_use.size) != n:
        raise ValueError("x0 length mismatch")

    m_use = 20 if m is None else int(m)
    if m_use <= 0:
        raise ValueError("m must be positive")
    k_use = None if k is None else int(k)
    if k_use is not None and k_use <= 0:
        raise ValueError("k must be positive when provided")

    t_solve0 = time.perf_counter()
    x, info = gcrotmk(
        A,
        b,
        x0=x0_use,
        maxiter=int(maxiter),
        M=M,
        callback=_cb,
        m=m_use,
        k=k_use,
        CU=recycle_space,
        atol=0.0,
        rtol=float(tol),
    )
    solve_time = time.perf_counter() - t_solve0
    x = np.asarray(x, dtype=np.float64).ravel()
    r = mv(x) - b
    resid = float(np.linalg.norm(r))
    rel = resid / bnorm if bnorm > 0.0 else resid
    out: dict[str, Any] = {
        "info": int(info),
        "niter": int(niter),
        "matvec_calls": int(matvec_calls),
        "matvec_time_total": float(matvec_time),
        "matvec_time_avg": float(matvec_time) / float(matvec_calls) if matvec_calls else 0.0,
        "solve_time_total": float(solve_time),
        "rhs_norm": float(bnorm),
        "residual_norm": resid,
        "residual_rel": float(rel),
        "solver": "gcrotmk",
    }
    return x, out


def _build_sa_lagrange_precond(
    *,
    n_orb: int,
    diag: np.ndarray,
    ci_ref_list: list[np.ndarray],
    ci_unflatten: Callable[[np.ndarray], Any],
    level_shift: float = 1e-8,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a CSF-compatible SA-CASSCF Lagrange preconditioner.

    This mirrors the logic of PySCF's `pyscf.grad.sacasscf.SACASLagPrec` but operates
    on flattened CI vectors (e.g. CSF/GUGA) rather than determinant-shaped (na,nb) CI arrays.

    Parameters
    ----------
    n_orb : int
        Number of orbital parameters.
    diag : np.ndarray
        Diagonal approximation of the Hessian.
    ci_ref_list : list[np.ndarray]
        Reference CI vectors.
    ci_unflatten : Callable
        Unflatten function for CI vectors.
    level_shift : float, optional
        Level shift for stability.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Preconditioner function.
    """

    diag = np.asarray(diag, dtype=np.float64).ravel()
    n_orb = int(n_orb)
    if n_orb < 0 or int(diag.size) < n_orb:
        raise ValueError("invalid n_orb/diag for SA preconditioner")

    ci_ref_flat = [np.asarray(c, dtype=np.float64).ravel() for c in ci_ref_list]
    sizes = {int(c.size) for c in ci_ref_flat}
    if len(sizes) != 1:
        raise ValueError("SA preconditioner requires consistent CI sizes across roots")
    nci = int(next(iter(sizes)))
    nroots = int(len(ci_ref_flat))
    if nroots <= 0:
        raise ValueError("empty ci_ref_list for SA preconditioner")

    d_orb = diag[:n_orb].copy()
    d_orb = d_orb + float(level_shift)
    d_orb[np.abs(d_orb) < 1e-8] = 1e-8
    r_orb = 1.0 / d_orb if d_orb.size else d_orb

    d_ci_list = ci_unflatten(diag[n_orb:])
    if not isinstance(d_ci_list, list) or len(d_ci_list) != nroots:
        raise ValueError("SA preconditioner expected list CI diag matching nroots")
    d_ci_flat = [np.asarray(v, dtype=np.float64).ravel() for v in d_ci_list]
    if any(int(v.size) != nci for v in d_ci_flat):
        raise ValueError("SA preconditioner CI diag size mismatch")

    cmat = np.stack(ci_ref_flat, axis=1)  # (nci, nroots)

    r_ci: list[np.ndarray] = []
    r_ci_sa: list[np.ndarray] = []
    for i in range(nroots):
        d_i = np.asarray(d_ci_flat[i], dtype=np.float64).copy()
        d_i = d_i + float(level_shift)
        d_i[np.abs(d_i) < 1e-8] = 1e-8
        r_i = 1.0 / d_i
        r_ci.append(r_i)

        rci_c = r_i[:, None] * cmat  # (nci,nroots)
        sci = cmat.T @ rci_c  # (nroots,nroots)
        try:
            sci_inv = np.linalg.inv(sci)
        except np.linalg.LinAlgError:  # pragma: no cover
            sci_inv = np.linalg.pinv(sci)
        r_ci_sa.append(rci_c @ sci_inv)  # (nci,nroots)

    def _precond(x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64).ravel()
        if int(xx.size) != int(diag.size):
            raise ValueError("SA preconditioner input length mismatch")

        x_orb = xx[:n_orb]
        x_ci = xx[n_orb:]

        out_orb = x_orb * r_orb if int(r_orb.size) else x_orb

        x_ci_list = ci_unflatten(x_ci)
        if not isinstance(x_ci_list, list) or len(x_ci_list) != nroots:
            raise ValueError("SA preconditioner expected list CI input")

        rx_mat = np.empty((nci, nroots), dtype=np.float64)
        for i in range(nroots):
            rx_mat[:, i] = r_ci[i] * np.asarray(x_ci_list[i], dtype=np.float64).ravel()

        sa_ovlp = cmat.T @ rx_mat  # (nroots,nroots)

        out_ci = np.empty_like(rx_mat)
        for i in range(nroots):
            out_ci[:, i] = rx_mat[:, i] - r_ci_sa[i] @ sa_ovlp[:, i]

        return np.concatenate([out_orb, out_ci.ravel(order="F")])

    return _precond


def build_mcscf_hessian_operator(
    mc,
    *,
    mo_coeff: np.ndarray | None = None,
    ci: Any | None = None,
    eris: Any | None = None,
    casdm1: np.ndarray | None = None,
    casdm2: np.ndarray | None = None,
    use_newton_hessian: bool | None = True,
) -> MCSCFHessianOp:
    """Build a reusable CASSCF super-Hessian linear operator for Z-vector solves.

    Parameters
    ----------
    mc : Any
        CASSCF object.
    mo_coeff : np.ndarray | None, optional
        MO coefficients.
    ci : Any | None, optional
        CI vector.
    eris : Any | None, optional
        Integral object.
    casdm1 : np.ndarray | None, optional
        Active 1-RDM.
    casdm2 : np.ndarray | None, optional
        Active 2-RDM.
    use_newton_hessian : bool | None, optional
        Whether to use exact Newton Hessian (default True).

    Returns
    -------
    MCSCFHessianOp
        Linear operator for the super-Hessian.
    """

    if mo_coeff is None:
        mo_coeff = getattr(mc, "mo_coeff", None)
    if ci is None:
        ci = getattr(mc, "ci", None)
    if mo_coeff is None or ci is None:
        raise ValueError("mo_coeff/ci must be provided or present on mc")

    ci_flat, ci_unflatten = _flatten_ci(ci)
    n_ci = int(ci_flat.size)
    if n_ci <= 0:
        raise ValueError("CI vector appears to be empty")

    is_sa = isinstance(ci, (list, tuple)) and int(len(ci)) > 1
    ci_ref_list: list[np.ndarray] | None = None
    sa_gram_inv: np.ndarray | None = None
    if is_sa:
        ci_ref_list = [np.asarray(c, dtype=np.float64).ravel() for c in ci]
        sizes = {int(c.size) for c in ci_ref_list}
        if len(sizes) == 1:
            nroots = int(len(ci_ref_list))
            gram = np.empty((nroots, nroots), dtype=np.float64)
            for i in range(nroots):
                ci_i = ci_ref_list[i]
                gram[i, i] = float(np.dot(ci_i, ci_i))
                for j in range(i + 1, nroots):
                    cij = float(np.dot(ci_i, ci_ref_list[j]))
                    gram[i, j] = cij
                    gram[j, i] = cij
            try:
                sa_gram_inv = np.linalg.inv(gram)
            except np.linalg.LinAlgError:  # pragma: no cover
                sa_gram_inv = np.linalg.pinv(gram)

    # Compute eris if not provided.
    _eris = eris
    if _eris is None and hasattr(mc, "ao2mo"):
        _eris = mc.ao2mo(mo_coeff)

    def _try_newton_gen_g_hop():
        if use_newton_hessian is False:
            return None
        try:
            import os  # noqa: PLC0415

            prefer = str(os.environ.get("CUGUGA_NEWTON_CASSCF", "internal")).strip().lower()
            if prefer in ("0", "false", "none", "off", "disable", "disabled"):
                return None

            from asuka.mcscf import newton_casscf as newton_casscf  # noqa: PLC0415

            impl = str(os.environ.get("CUGUGA_NEWTON_CASSCF_IMPL", "internal")).strip().lower()
            if prefer in ("internal",):
                impl = "internal"
            if impl not in ("internal", "cuguga", "owned"):
                raise ValueError("CUGUGA_NEWTON_CASSCF_IMPL must be one of: internal")

            def _gen_g_hop(mc, mo, ci, eris, verbose=0):
                return newton_casscf.gen_g_hop(mc, mo, ci, eris, verbose=verbose, implementation=impl)

            return _gen_g_hop
        except Exception:
            if use_newton_hessian:
                raise
            return None

    gen_newton = _try_newton_gen_g_hop()
    if gen_newton is not None:
        try:
            g_all, _g_update, h_op, h_diag = gen_newton(mc, mo_coeff, ci, _eris, verbose=0)
            g_all = np.asarray(g_all, dtype=np.float64).ravel()
            n_tot = int(g_all.size)
            n_orb = n_tot - n_ci
            if n_orb < 0:
                raise RuntimeError(
                    "newton_casscf.gen_g_hop returned inconsistent dimensions: "
                    f"n_tot={n_tot} < n_ci={n_ci}"
                )

            # Detect GPU mode from h_op's closure (eris on GPU → full GPU matvec).
            _mv_gpu_mode = False
            try:
                import cupy as _cp_mv  # type: ignore[import-not-found]
                # Check if eris are on GPU by inspecting the integral object.
                _eris_ppaa = getattr(_eris, "ppaa", None) if _eris is not None else None
                if _cp_mv is not None and _eris_ppaa is not None and isinstance(_eris_ppaa, _cp_mv.ndarray):
                    _mv_gpu_mode = True
                    # Pre-upload SA reference CI vectors and gram_inv for GPU projection.
                    if is_sa and ci_ref_list is not None:
                        _ci_ref_dev = [_cp_mv.asarray(c, dtype=_cp_mv.float64).ravel() for c in ci_ref_list]
                        _sa_gram_inv_dev = _cp_mv.asarray(sa_gram_inv, dtype=_cp_mv.float64) if sa_gram_inv is not None else None
                        _ci_sizes_mv = [int(c.size) for c in _ci_ref_dev]
                        _ci_offs_mv: list[int] = [0]
                        for _ss in _ci_sizes_mv[:-1]:
                            _ci_offs_mv.append(_ci_offs_mv[-1] + _ss)
            except Exception:
                _cp_mv = None

            def _ci_unflatten_xp(xp_mod, flat):
                """Unflatten CI vector using xp — works for both numpy and CuPy."""
                if not is_sa or ci_ref_list is None:
                    return [flat.copy()]
                parts = []
                off = 0
                for c in (ci_ref_list if xp_mod is np else _ci_ref_dev):
                    s = int(c.size)
                    parts.append(flat[off:off + s].copy())
                    off += s
                return parts

            def _mv(x):
                # Detect xp from input — CuPy GMRES passes CuPy arrays.
                _is_gpu = _cp_mv is not None and isinstance(x, _cp_mv.ndarray)
                xp = _cp_mv if _is_gpu else np
                x = xp.asarray(x, dtype=xp.float64).ravel()
                if x.size != n_tot:
                    raise ValueError("h_op input length mismatch")

                if is_sa:
                    x_orb = x[:n_orb]
                    x_ci_list = _ci_unflatten_xp(xp, x[n_orb:])
                    if ci_ref_list is None:
                        raise RuntimeError("internal error: expected ci_ref_list for SA-CASSCF")
                    _refs = _ci_ref_dev if _is_gpu else ci_ref_list
                    _ginv = _sa_gram_inv_dev if (_is_gpu and _mv_gpu_mode) else sa_gram_inv
                    x_ci_list = _project_sa_ci_components(_refs, x_ci_list, gram_inv=_ginv)
                    x = xp.concatenate([x_orb, xp.concatenate([xp.asarray(v, dtype=xp.float64).ravel() for v in x_ci_list])])

                y = h_op(x)
                y = xp.asarray(y, dtype=xp.float64).ravel()
                if y.size != n_tot:
                    raise ValueError("h_op output length mismatch")

                if is_sa:
                    y_orb = y[:n_orb]
                    y_ci_list = _ci_unflatten_xp(xp, y[n_orb:])
                    if ci_ref_list is None:
                        raise RuntimeError("internal error: expected ci_ref_list for SA-CASSCF")
                    y_ci_list = _project_sa_ci_components(_refs, y_ci_list, gram_inv=_ginv)
                    y = xp.concatenate([y_orb, xp.concatenate([xp.asarray(v, dtype=xp.float64).ravel() for v in y_ci_list])])

                return y

            diag = None
            try:
                diag = _extract_h_diag(h_diag, n_orb, ci)
            except Exception:
                diag = None
            if diag is not None and int(np.asarray(diag).size) != n_tot:
                diag = None

            return MCSCFHessianOp(
                mv=_mv,
                diag=diag,
                n_orb=int(n_orb),
                n_ci=int(n_ci),
                ci_template=ci,
                ci_unflatten=ci_unflatten,
                orb_only=False,
                is_sa=bool(is_sa),
                ci_ref_list=ci_ref_list,
                sa_gram_inv=sa_gram_inv,
                gpu_mode=bool(_mv_gpu_mode),
            )
        except Exception:
            if use_newton_hessian:
                raise
            gen_newton = None  # fall back

    # Fallback: orbital-only system from mc1step.gen_g_hop
    if not hasattr(mc, "gen_g_hop"):
        raise AttributeError("mc object does not provide gen_g_hop")

    # Compute casdm1/casdm2 if not provided.
    _casdm1 = casdm1
    _casdm2 = casdm2
    if _casdm1 is None or _casdm2 is None:
        if hasattr(mc, "fcisolver") and hasattr(mc.fcisolver, "make_rdm12"):
            _dm1, _dm2 = mc.fcisolver.make_rdm12(ci, mc.ncas, mc.nelecas)
            if _casdm1 is None:
                _casdm1 = _dm1
            if _casdm2 is None:
                _casdm2 = _dm2
        else:
            raise RuntimeError("Cannot compute casdm1/casdm2; provide them explicitly")

    # Compute eris if not provided.
    if _eris is None and hasattr(mc, "ao2mo"):
        _eris = mc.ao2mo(mo_coeff)

    try:
        out = mc.gen_g_hop(mo_coeff, ci, _casdm1, _casdm2, _eris)
    except TypeError:
        try:
            out = mc.gen_g_hop(mo_coeff, ci, casdm1=_casdm1, casdm2=_casdm2, eris=_eris)
        except TypeError as e:
            raise RuntimeError(f"Failed to call mc.gen_g_hop: {e}") from e

    if not isinstance(out, (list, tuple)) or len(out) != 4:
        raise RuntimeError(
            "Unsupported gen_g_hop return signature. Expected 4-tuple "
            "(g_orb, g_ci, h_op, h_diag)."
        )
    g_orb, _g_ci, h_op, h_diag = out
    g_orb = np.asarray(g_orb, dtype=np.float64).ravel()
    n_orb = int(g_orb.size)
    if n_orb < 0:
        raise RuntimeError("gen_g_hop returned invalid orbital gradient size")

    def _orb_mv(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != n_orb:
            raise ValueError("h_op input length mismatch")
        y = h_op(x)
        y = np.asarray(y, dtype=np.float64).ravel()
        if y.size != n_orb:
            raise ValueError("h_op output length mismatch")
        return y

    # Orbital-only diagonal preconditioner (best-effort).
    diag_orb = None
    if h_diag is not None:
        if isinstance(h_diag, np.ndarray):
            d = np.asarray(h_diag, dtype=np.float64).ravel()
            if int(d.size) == n_orb:
                diag_orb = d
        elif isinstance(h_diag, (list, tuple)) and len(h_diag) == 2:
            d_orb = np.asarray(h_diag[0], dtype=np.float64).ravel()
            if int(d_orb.size) == n_orb:
                diag_orb = d_orb

    return MCSCFHessianOp(
        mv=_orb_mv,
        diag=diag_orb,
        n_orb=int(n_orb),
        n_ci=int(n_ci),
        ci_template=ci,
        ci_unflatten=ci_unflatten,
        orb_only=True,
        is_sa=bool(is_sa),
        ci_ref_list=ci_ref_list,
        sa_gram_inv=sa_gram_inv,
    )


def solve_mcscf_zvector(
    mc,
    *,
    mo_coeff: np.ndarray | None = None,
    ci: Any | None = None,
    rhs_orb: np.ndarray | None = None,
    rhs_ci: Any | None = None,
    eris: Any | None = None,
    casdm1: np.ndarray | None = None,
    casdm2: np.ndarray | None = None,
    tol: float = 1e-10,
    maxiter: int = 200,
    use_newton_hessian: bool | None = True,
    restart: int | None = None,
    x0: np.ndarray | None = None,
    method: str = "gmres",
    gcrotmk_k: int | None = None,
    recycle_space: list[tuple[np.ndarray | None, np.ndarray]] | None = None,
    hessian_op: MCSCFHessianOp | None = None,
    enforce_absorb_h1e_direct: bool = True,
    project_sa_rhs: bool = True,
    auto_rdm_backend_cuda: bool = True,
    rdm_cuda_threshold_ncsf: int = 4096,
) -> MCSCFZVectorResult:
    """Solve the MCSCF/CASSCF Z-vector linear system using PySCF's Hessian operator.

    Parameters
    ----------
    mc : Any
        CASSCF object.
    mo_coeff : np.ndarray | None, optional
        MO coefficients.
    ci : Any | None, optional
        CI vector.
    rhs_orb : np.ndarray | None, optional
        Orbital part of RHS.
    rhs_ci : Any | None, optional
        CI part of RHS.
    eris : Any | None, optional
        Integral object.
    casdm1 : np.ndarray | None, optional
        Active 1-RDM.
    casdm2 : np.ndarray | None, optional
        Active 2-RDM.
    tol : float, optional
        Convergence tolerance.
    maxiter : int, optional
        Max iterations.
    use_newton_hessian : bool | None, optional
        Use exact Newton Hessian (default True).
    restart : int | None, optional
        Restart size (GMRES).
    x0 : np.ndarray | None, optional
        Initial guess.
    method : str, optional
        Solver method ('gmres' or 'gcrotmk').
    gcrotmk_k : int | None, optional
        Number of vectors to keep (GCROT(m,k)).
    recycle_space : list | None, optional
        Recycle space for GCROT(m,k).
    hessian_op : MCSCFHessianOp | None, optional
        Pre-built Hessian operator.
    enforce_absorb_h1e_direct : bool, optional
        Force 'direct' mode for absorb_h1e.
    project_sa_rhs : bool, optional
        Project SA RHS (default True).
    auto_rdm_backend_cuda : bool, optional
        Auto-enable CUDA RDM backend.
    rdm_cuda_threshold_ncsf : int, optional
        Threshold for CUDA RDM backend.

    Returns
    -------
    MCSCFZVectorResult
        Solver result.

    Notes
    -----
    - Uses PySCF's super-Hessian action (`mc.gen_g_hop` / `newton_casscf.gen_g_hop`) and solves
      the linear system with GMRES or GCROT(m,k).
    - For SA-CASSCF, the CI part of the RHS is optionally projected to the orthogonal complement
      of the SA CI root span. This matches the gauge used by PySCF's SA Lagrange machinery.
    - For SA-CASSCF with CSF/GUGA CI vectors, the default diagonal preconditioner can inject
      root-span components. When a diagonal is available, we switch to a CSF-compatible analogue
      of PySCF's `SACASLagPrec` to improve stability.
    """

    fcisolver = getattr(mc, "fcisolver", None)
    if enforce_absorb_h1e_direct:
        ctx_absorb = _maybe_set_attr(fcisolver, "absorb_h1e_mode", "direct")
    else:
        ctx_absorb = nullcontext(False)

    def _want_cuda_rdm(ncsf: int) -> bool:
        if not bool(auto_rdm_backend_cuda):
            return False
        if ncsf < int(rdm_cuda_threshold_ncsf):
            return False
        if fcisolver is None:
            return False
        try:
            cur = str(getattr(fcisolver, "rdm_backend", "auto")).strip().lower()
        except Exception:
            return False
        if cur != "auto":
            return False
        try:
            from asuka.cuda.cuda_backend import has_cuda_ext  # noqa: PLC0415

            if not has_cuda_ext():
                return False
            import cupy as cp  # type: ignore[import-not-found]  # noqa: F401
        except Exception:
            return False
        return True

    with ctx_absorb as absorb_direct_changed:
        if hessian_op is not None:
            if mo_coeff is not None or ci is not None or eris is not None or casdm1 is not None or casdm2 is not None:
                raise ValueError("When hessian_op is provided, do not pass mo_coeff/ci/eris/casdm1/casdm2")
            op = hessian_op
        else:
            op = build_mcscf_hessian_operator(
                mc,
                mo_coeff=mo_coeff,
                ci=ci,
                eris=eris,
                casdm1=casdm1,
                casdm2=casdm2,
                use_newton_hessian=use_newton_hessian,
            )

        # Heuristic: for large CI spaces, switching only the RDM builder to CUDA can
        # dramatically reduce per-matvec time (trans_rdm12 dominates many Z-solves).
        nroots_hint = int(len(op.ci_ref_list)) if op.is_sa and op.ci_ref_list is not None else 1
        ncsf_hint = int(op.n_ci) // max(nroots_hint, 1)
        if (not bool(op.orb_only)) and _want_cuda_rdm(int(ncsf_hint)):
            ctx_rdm = _maybe_set_attr(fcisolver, "rdm_backend", "cuda")
        else:
            ctx_rdm = nullcontext(False)

        method_l = str(method).strip().lower()
        if method_l not in ("gmres", "gcrotmk"):
            raise ValueError("method must be 'gmres' or 'gcrotmk'")

        # Orbital RHS
        if rhs_orb is None:
            rhs_orb_flat = np.zeros(int(op.n_orb), dtype=np.float64)
        else:
            rhs_orb_flat = np.asarray(rhs_orb, dtype=np.float64).ravel()
            if int(rhs_orb_flat.size) != int(op.n_orb):
                raise ValueError("rhs_orb size mismatch")

        # CI RHS
        if rhs_ci is None:
            rhs_ci_flat = np.zeros(int(op.n_ci), dtype=np.float64)
        else:
            rhs_ci_flat, _ = _flatten_ci(rhs_ci)
            if int(rhs_ci_flat.size) != int(op.n_ci):
                raise ValueError("rhs_ci size mismatch")

        if op.is_sa and bool(project_sa_rhs):
            rhs_ci_list = op.ci_unflatten(rhs_ci_flat)
            if not isinstance(rhs_ci_list, list) or op.ci_ref_list is None:
                raise RuntimeError("internal error: expected list CI unpacking for SA-CASSCF")
            rhs_ci_list = _project_sa_ci_components(op.ci_ref_list, rhs_ci_list, gram_inv=op.sa_gram_inv)
            rhs_ci_flat = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in rhs_ci_list])

        diag_use: np.ndarray | None = op.diag
        precond_use: Callable[[np.ndarray], np.ndarray] | None = None
        if op.is_sa and (not bool(op.orb_only)) and op.diag is not None and op.ci_ref_list is not None:
            try:
                precond_use = _build_sa_lagrange_precond(
                    n_orb=int(op.n_orb),
                    diag=np.asarray(op.diag, dtype=np.float64),
                    ci_ref_list=op.ci_ref_list,
                    ci_unflatten=op.ci_unflatten,
                )
                diag_use = None
            except Exception:
                precond_use = None
                diag_use = op.diag

        info: dict[str, Any]
        z: np.ndarray
        z_orb: np.ndarray
        z_ci: Any

        if op.orb_only:
            if int(op.n_orb) == 0:
                z_orb = np.zeros(0, dtype=np.float64)
                z = np.concatenate([z_orb, np.zeros(int(op.n_ci), dtype=np.float64)])
                z_ci = op.ci_unflatten(np.zeros(int(op.n_ci), dtype=np.float64))
                info = {
                    "info": 0,
                    "niter": 0,
                    "matvec_calls": 0,
                    "rhs_norm": 0.0,
                    "residual_norm": 0.0,
                    "residual_rel": 0.0,
                }
            else:
                x0_orb = None
                if x0 is not None:
                    x0_flat = np.asarray(x0, dtype=np.float64).ravel()
                    if int(x0_flat.size) == int(op.n_tot):
                        x0_orb = x0_flat[: int(op.n_orb)].copy()
                    elif int(x0_flat.size) == int(op.n_orb):
                        x0_orb = x0_flat.copy()
                    else:
                        raise ValueError("x0 length mismatch")

                b_orb = -rhs_orb_flat
                if method_l == "gcrotmk":
                    with ctx_rdm:
                        z_orb, info = _gcrotmk_solve(
                            op.mv,
                            b_orb,
                            diag_precond=diag_use,
                            precond=precond_use,
                            tol=float(tol),
                            maxiter=int(maxiter),
                            m=restart,
                            k=gcrotmk_k,
                            x0=x0_orb,
                            recycle_space=recycle_space,
                        )
                else:
                    gmres_kwargs: dict[str, Any] = {
                        "diag_precond": diag_use,
                        "precond": precond_use,
                        "tol": float(tol),
                        "maxiter": int(maxiter),
                    }
                    if restart is not None:
                        gmres_kwargs["restart"] = int(restart)
                    if x0_orb is not None:
                        gmres_kwargs["x0"] = np.asarray(x0_orb, dtype=np.float64).ravel()
                    with ctx_rdm:
                        z_orb, info = _gmres_solve(op.mv, b_orb, **gmres_kwargs, use_gpu=False)

                z_orb = np.asarray(z_orb, dtype=np.float64).ravel()
                z = np.concatenate([z_orb, np.zeros(int(op.n_ci), dtype=np.float64)])
                z_ci = op.ci_unflatten(np.zeros(int(op.n_ci), dtype=np.float64))
        else:
            rhs = np.concatenate([rhs_orb_flat, rhs_ci_flat])

            x0_use = None
            if x0 is not None:
                x0_use = np.asarray(x0, dtype=np.float64).ravel()
                if int(x0_use.size) != int(op.n_tot):
                    raise ValueError("x0 length mismatch")
                if op.is_sa:
                    x0_orb = x0_use[: int(op.n_orb)]
                    x0_ci_list = op.ci_unflatten(x0_use[int(op.n_orb) :])
                    if not isinstance(x0_ci_list, list) or op.ci_ref_list is None:
                        raise RuntimeError("internal error: expected list CI unpacking for SA-CASSCF")
                    x0_ci_list = _project_sa_ci_components(op.ci_ref_list, x0_ci_list, gram_inv=op.sa_gram_inv)
                    x0_use = np.concatenate(
                        [x0_orb, np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in x0_ci_list])]
                    )

            b = -rhs
            if method_l == "gcrotmk":
                with ctx_rdm:
                    z, info = _gcrotmk_solve(
                        op.mv,
                        b,
                        diag_precond=diag_use,
                        precond=precond_use,
                        tol=float(tol),
                        maxiter=int(maxiter),
                        m=restart,
                        k=gcrotmk_k,
                        x0=x0_use,
                        recycle_space=recycle_space,
                    )
            else:
                gmres_kwargs = {
                    "diag_precond": diag_use,
                    "precond": precond_use,
                    "tol": float(tol),
                    "maxiter": int(maxiter),
                }
                if restart is not None:
                    gmres_kwargs["restart"] = int(restart)
                if x0_use is not None:
                    gmres_kwargs["x0"] = np.asarray(x0_use, dtype=np.float64).ravel()
                with ctx_rdm:
                    z, info = _gmres_solve(op.mv, b, **gmres_kwargs, use_gpu=bool(op.gpu_mode))

            z = np.asarray(z, dtype=np.float64).ravel()
            z_orb = z[: int(op.n_orb)].copy()
            z_ci = op.ci_unflatten(z[int(op.n_orb) :])

        resid = float(info.get("residual_norm", np.nan))
        rel = float(info.get("residual_rel", np.nan))
        solver_info = int(info.get("info", 1))
        if np.isfinite(rel):
            converged = bool(rel <= float(tol))
        else:  # pragma: no cover
            converged = bool(solver_info == 0)

        niter = int(info.get("niter", 0))
        if niter <= 0 and solver_info > 0:
            niter = int(solver_info)

        info_out = dict(info)
        info_out["absorb_h1e_direct"] = bool(absorb_direct_changed)
        if bool(op.orb_only):
            # `mc1step.gen_g_hop` convention.
            info_out.setdefault("hessian_backend", "mc1step")
        else:
            # `newton_casscf.gen_g_hop` convention (PySCF-style packed vector).
            info_out.setdefault("hessian_backend", "newton")

        return MCSCFZVectorResult(
            converged=converged,
            niter=niter,
            residual_norm=resid,
            z_orb=np.asarray(z_orb, dtype=np.float64).ravel(),
            z_ci=z_ci,
            z_packed=np.asarray(z, dtype=np.float64).ravel(),
            info=info_out,
        )


def build_ci_gradient_from_effective_integrals(
    fcisolver,
    *,
    h1_eff: np.ndarray,
    h2_eff: np.ndarray,
    ci0: Any,
    norb: int,
    nelec: int | tuple[int, int],
    normalize: bool = True,
) -> Any:
    """Form the CI-space gradient (RHS) from an effective Hamiltonian.

    This helper is useful when an external energy contribution (e.g. CASPT2)
    has been backpropagated to derivatives w.r.t. the active-space RDMs,
    yielding an operator of the form

    .. math::

        F = \\sum_{pq} h1_{\\text{eff}}[p,q] E_{pq} + \\frac{1}{2} \\sum_{pqrs} h2_{\\text{eff}}[p,q,r,s] E_{pqrs}

    The gradient w.r.t. CI coefficients (for a *normalized, real* CI vector) is

    .. math::

        g_{\\text{ci}} = (F + F^\\dagger) |ci\\rangle - (\\langle ci | (F + F^\\dagger) | ci \\rangle) |ci\\rangle

    Parameters
    ----------
    fcisolver : Any
        A PySCF FCI-solver object (e.g. :class:`asuka.solver.GUGAFCISolver`).
    h1_eff : np.ndarray
        Effective one-electron integrals in active MO basis.
    h2_eff : np.ndarray
        Effective two-electron integrals in active MO basis (physicist notation).
    ci0 : Any
        Reference CI vector(s).
    norb : int
        Number of active orbitals.
    nelec : int | tuple[int, int]
        Number of active electrons.
    normalize : bool, optional
        If True, project out the component along ``ci0`` (default True).

    Returns
    -------
    Any
        CI gradient vector(s).
    """

    h1_eff = np.asarray(h1_eff, dtype=np.float64)
    h2_eff = np.asarray(h2_eff, dtype=np.float64)

    # Multi-root support (simple, independent states).
    if isinstance(ci0, (list, tuple)):
        out: list[np.ndarray] = []
        for v in ci0:
            out.append(
                build_ci_gradient_from_effective_integrals(
                    fcisolver,
                    h1_eff=h1_eff,
                    h2_eff=h2_eff,
                    ci0=v,
                    norb=norb,
                    nelec=nelec,
                    normalize=normalize,
                )
            )
        return out

    c0 = np.asarray(ci0, dtype=np.float64).ravel()
    hc = np.asarray(fcisolver.contract_2e(h2_eff, c0, int(norb), nelec, h1e=h1_eff), dtype=np.float64).ravel()
    if hc.size != c0.size:
        raise ValueError("contract_2e returned unexpected CI vector length")

    # For general (not-necessarily symmetric) effective integrals, only the Hermitian part of the
    # induced CI-space operator contributes to a real energy functional. Use (F + F†)|ci>.
    h1_dag = h1_eff.T
    h2_dag = h2_eff.transpose(3, 2, 1, 0)
    hc_dag = np.asarray(
        fcisolver.contract_2e(h2_dag, c0, int(norb), nelec, h1e=h1_dag), dtype=np.float64
    ).ravel()
    if hc_dag.size != c0.size:
        raise ValueError("contract_2e returned unexpected CI vector length (dagger path)")

    g = hc + hc_dag
    if normalize:
        denom = float(np.dot(c0, c0))
        if denom > 0.0:
            alpha = float(np.dot(c0, g)) / denom
            g = g - alpha * c0
    return g


def effective_active_rdms_from_ci_zvector(
    fcisolver,
    *,
    ci0: Any,
    z_ci: Any,
    norb: int,
    nelec: int | tuple[int, int],
    weights: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the CI-response contribution to the active-space 1- and 2-RDMs.

    For a real wavefunction, the first-order change in the RDMs due to a CI
    response vector ``z_ci`` is

    .. math::

        dD = 2 \\langle z | E | 0 \\rangle \\\\
        d\\Gamma = 2 \\langle z | E_2 | 0 \\rangle

    where ``<z|`` is the bra corresponding to the Z-vector solution in CI space.

    Parameters
    ----------
    fcisolver : Any
        FCI solver object.
    ci0 : Any
        Reference CI vector.
    z_ci : Any
        CI Z-vector.
    norb : int
        Number of active orbitals.
    nelec : int | tuple[int, int]
        Number of active electrons.
    weights : Sequence[float] | None, optional
        Weights for state-averaged RDM.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Effective active-space (1-RDM, 2-RDM).

    Notes
    -----
    This routine uses the solver's ``trans_rdm12`` implementation, which is
    available for :class:`asuka.solver.GUGAFCISolver`.
    """

    def _trans_rdm12_single(ci_bra: np.ndarray, ci_ket: np.ndarray):
        """Call a single-root trans_rdm12 even when `fcisolver` is state-average wrapped.

        PySCF's `state_average_` wraps the solver in a `StateAverage*` subclass which
        overrides `trans_rdm12` to accept *lists of CI vectors*.  For Z-vector → RDM
        response we need the underlying single-root transition RDMs between two CI
        vectors.  When available, bypass the wrapper by dispatching to the base class
        stored in `fcisolver._base_class`.
        """

        base_cls = getattr(fcisolver, "_base_class", None)
        if base_cls is not None and base_cls is not type(fcisolver):
            base_trans = getattr(base_cls, "trans_rdm12", None)
            if base_trans is not None:
                return base_trans(fcisolver, ci_bra, ci_ket, int(norb), nelec)
        return fcisolver.trans_rdm12(ci_bra, ci_ket, int(norb), nelec)

    # Multi-root: return the (optionally weighted) sum over roots. This is the
    # natural object needed for building effective densities for orbital-response
    # contractions, as long as the Z-vector solve was done in the same multi-root
    # parameter space.
    if isinstance(ci0, (list, tuple)) or isinstance(z_ci, (list, tuple)):
        if not isinstance(ci0, (list, tuple)) or not isinstance(z_ci, (list, tuple)):
            raise ValueError("ci0 and z_ci must both be list/tuple for multi-root")
        if len(ci0) != len(z_ci):
            raise ValueError("ci0 and z_ci must have the same number of roots")
        nroots = int(len(ci0))
        if weights is None:
            weights = [1.0] * nroots
        if len(weights) != nroots:
            raise ValueError("weights must have length nroots")

        dm1_acc = np.zeros((int(norb), int(norb)), dtype=np.float64)
        dm2_acc = np.zeros((int(norb), int(norb), int(norb), int(norb)), dtype=np.float64)
        for w, c0_i, z_i in zip(weights, ci0, z_ci):
            c0 = np.asarray(c0_i, dtype=np.float64).ravel()
            z = np.asarray(z_i, dtype=np.float64).ravel()
            dm1_t, dm2_t = _trans_rdm12_single(z, c0)
            dm1_acc += float(w) * (2.0 * np.asarray(dm1_t, dtype=np.float64))
            dm2_acc += float(w) * (2.0 * np.asarray(dm2_t, dtype=np.float64))
        return dm1_acc, dm2_acc

    c0 = np.asarray(ci0, dtype=np.float64).ravel()
    z = np.asarray(z_ci, dtype=np.float64).ravel()

    dm1_t, dm2_t = _trans_rdm12_single(z, c0)
    dm1_t = np.asarray(dm1_t, dtype=np.float64)
    dm2_t = np.asarray(dm2_t, dtype=np.float64)
    w = 1.0 if weights is None else float(np.asarray(weights, dtype=np.float64).ravel()[0])
    return (2.0 * w) * dm1_t, (2.0 * w) * dm2_t


def build_orbital_gradient_from_effective_densities(
    *,
    h1_mo: np.ndarray,
    eri_mo: np.ndarray,
    d1_mo: np.ndarray,
    d2_mo: np.ndarray,
) -> np.ndarray:
    """Compute an orbital-rotation gradient matrix from effective densities.

    Many post-CASSCF Lagrangian formalisms produce *effective* one- and
    two-particle density-like objects (often denoted :math:`\\bar{D}, \\bar{\\Gamma}`) such
    that the first-order change in the target energy w.r.t. the MO integrals is

    .. math::

        dE = \\sum_{pq} h1_{\\text{mo}}[p,q] d1_{\\text{mo}}[p,q]
           + \\frac{1}{2} \\sum_{pqrs} eri_{\\text{mo}}[p,q,r,s] d2_{\\text{mo}}[p,q,r,s]

    Treating this as an expectation value-like functional, the corresponding
    orbital rotation gradient can be written in terms of the antisymmetric part
    of a "generalized Fock" matrix

    .. math::

        K[p,q] = \\sum_r h1_{\\text{mo}}[p,r] d1_{\\text{mo}}[r,q]
               + \\sum_{r,s,t} eri_{\\text{mo}}[p,r,s,t] d2_{\\text{mo}}[r,q,s,t]

        g_{\\text{orb}}[p,q] = 2 (K[p,q] - K[q,p])

    Parameters
    ----------
    h1_mo : np.ndarray
        Effective 1-electron integrals (MO basis).
    eri_mo : np.ndarray
        Effective 2-electron integrals (MO basis).
    d1_mo : np.ndarray
        Effective 1-RDM.
    d2_mo : np.ndarray
        Effective 2-RDM.

    Returns
    -------
    np.ndarray
        Antisymmetric orbital gradient matrix ``g_orb``.

    Notes
    -----
    * This routine is *tensor-form* and therefore intended for small MO spaces
      (e.g. active-space-only orbital problems, debugging, or toy systems).
      For production gradients, you typically avoid materializing full
      `(nmo,nmo,nmo,nmo)` ERI tensors.
    """

    h1_mo = np.asarray(h1_mo, dtype=np.float64)
    eri_mo = np.asarray(eri_mo, dtype=np.float64)
    d1_mo = np.asarray(d1_mo, dtype=np.float64)
    d2_mo = np.asarray(d2_mo, dtype=np.float64)

    if h1_mo.ndim != 2 or h1_mo.shape[0] != h1_mo.shape[1]:
        raise ValueError("h1_mo must be square (nmo,nmo)")
    nmo = int(h1_mo.shape[0])
    if d1_mo.shape != (nmo, nmo):
        raise ValueError("d1_mo must have shape (nmo,nmo)")
    if eri_mo.shape != (nmo, nmo, nmo, nmo):
        raise ValueError("eri_mo must have shape (nmo,nmo,nmo,nmo)")
    if d2_mo.shape != (nmo, nmo, nmo, nmo):
        raise ValueError("d2_mo must have shape (nmo,nmo,nmo,nmo)")

    k1 = np.einsum("pr,rq->pq", h1_mo, d1_mo, optimize=True)
    k2 = np.einsum("prst,rqst->pq", eri_mo, d2_mo, optimize=True)
    k = k1 + k2
    g = 2.0 * (k - k.T)
    # Enforce antisymmetry numerically.
    return 0.5 * (g - g.T)


def pack_orbital_gradient(mc: Any, g_mat: np.ndarray) -> np.ndarray:
    """Pack an orbital gradient matrix into PySCF's "uniq var" vector.

    PySCF's CASSCF machinery represents orbital rotations in a packed format
    (unique, non-redundant variables).  If the given `mc` object provides
    ``pack_uniq_var``, this helper uses it.

    If not available, this falls back to packing the strict lower triangle of
    `g_mat` in row-major order.

    Parameters
    ----------
    mc : Any
        PySCF MC object (optional).
    g_mat : np.ndarray
        Antisymmetric gradient matrix (nmo,nmo).

    Returns
    -------
    np.ndarray
        Packed gradient vector.
    """

    g_mat = np.asarray(g_mat, dtype=np.float64)
    if g_mat.ndim != 2 or g_mat.shape[0] != g_mat.shape[1]:
        raise ValueError("g_mat must be square")

    fn = getattr(mc, "pack_uniq_var", None)
    if callable(fn):
        v = fn(g_mat)
        return np.asarray(v, dtype=np.float64).ravel()

    # Fallback: strict lower triangle.
    n = int(g_mat.shape[0])
    idx = np.tril_indices(n, k=-1)
    return np.asarray(g_mat[idx], dtype=np.float64)
