r"""Preconditioned conjugate gradient solver for CASPT2 equations.

Ports OpenMolcas ``pcg.f``.

Mathematical Problem
--------------------
Solves the CASPT2 amplitude equations:

.. math::

    (\hat{H}_0 - E_0)\,|T\rangle = -|V\rangle

for the first-order amplitudes :math:`T_P`, where:

- :math:`\hat{H}_0` is Dyall's zeroth-order Hamiltonian
- :math:`E_0` is the zeroth-order energy (EASUM + inactive contribution)
- :math:`|V\rangle` is the RHS coupling vector
  :math:`V_P = \langle\Phi_P|\hat{H}|0\rangle`

Solver Strategies
~~~~~~~~~~~~~~~~~
**Direct divide** (``pcg_solve``):
When all off-diagonal Fock blocks are zero (quasi-canonical orbitals),
:math:`\hat{H}_0` is diagonal in the SR basis and the solution is:

.. math::

    T_P = \frac{-V_P}{d_P}

where :math:`d_P = b_\mu + \varepsilon^{\text{ext}}` is the per-function
denominator.

**Iterative PCG** (``pcg_solve_iterative``):
When inter-case Fock couplings exist, the system is solved iteratively
using preconditioned conjugate gradients.  The diagonal of
:math:`\hat{H}_0 - E_0` serves as the preconditioner:

.. math::

    M^{-1} r_k = z_k, \qquad M_{PP} = d_P

The PCG iteration follows the standard scheme:

1. :math:`r_0 = b - A x_0`, :math:`z_0 = M^{-1} r_0`, :math:`p_0 = z_0`
2. :math:`\alpha_k = (r_k \cdot z_k) / (p_k \cdot A p_k)`
3. :math:`x_{k+1} = x_k + \alpha_k\, p_k`
4. :math:`r_{k+1} = r_k - \alpha_k\, A p_k`
5. :math:`z_{k+1} = M^{-1} r_{k+1}`,
   :math:`\beta_k = (r_{k+1} \cdot z_{k+1}) / (r_k \cdot z_k)`
6. :math:`p_{k+1} = z_{k+1} + \beta_k\, p_k`
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PCGResult:
    """Result of PCG solver."""

    converged: bool
    niter: int
    residual: float
    amplitudes: list[np.ndarray]
    info: dict = field(default_factory=dict)


def pcg_solve(
    h0_diag: list[np.ndarray],
    rhs: list[np.ndarray],
    *,
    tol: float = 1e-8,
    maxiter: int = 200,
    verbose: int = 0,
) -> PCGResult:
    """Solve the CASPT2 linear equations using preconditioned conjugate gradient.

    In the diagonalized S/B basis, H0 is diagonal, so the equations become:
        (H0_diag - E0) * T = -V
    which can be solved directly (no iteration needed).

    For the imaginary-shift case or when H0 is not perfectly diagonal,
    a PCG iteration would be needed. Here we implement the direct solution
    first and fall back to iterative for non-diagonal cases.

    Parameters
    ----------
    h0_diag : list of arrays
        Diagonal of (H0 - E0) per case, in the diagonalized basis.
    rhs : list of arrays
        RHS vectors per case (already transformed to diag basis).
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum number of iterations.
    verbose : int
        Verbosity level.

    Returns
    -------
    PCGResult with solution amplitudes.
    """
    amplitudes = []
    total_residual = 0.0

    for case_idx, (diag, v) in enumerate(zip(h0_diag, rhs)):
        if v.size == 0 or diag.size == 0:
            amplitudes.append(np.zeros_like(v))
            continue

        # Direct solution: T = -V / (H0_diag - E0)
        # Guard against near-zero denominators (intruder states)
        t = np.zeros_like(v)
        mask = np.abs(diag) > 1e-14
        t[mask] = -v[mask] / diag[mask]

        # Compute residual: r = (H0 - E0)*T + V
        residual = diag * t + v
        res_norm = float(np.linalg.norm(residual))
        total_residual = max(total_residual, res_norm)

        if verbose >= 2:
            e_case = float(np.dot(t.ravel(), v.ravel()))
            print(f"  Case {case_idx + 1}: nfun={v.size}, E2={e_case:.10f}, |r|={res_norm:.2e}")

        amplitudes.append(t)

    converged = total_residual < tol

    return PCGResult(
        converged=converged,
        niter=1,
        residual=total_residual,
        amplitudes=amplitudes,
        info={"method": "direct"},
    )


def pcg_solve_iterative(
    sigma_op,
    rhs: list[np.ndarray],
    h0_diag: list[np.ndarray],
    *,
    tol: float = 1e-8,
    maxiter: int = 200,
    verbose: int = 0,
) -> PCGResult:
    """Iterative PCG solver for when H0 is not diagonal.

    Uses the diagonal of H0 as preconditioner.
    """
    # Flatten all cases into a single vector
    sizes = [v.size for v in rhs]
    total_size = sum(sizes)
    if total_size == 0:
        return PCGResult(
            converged=True, niter=0, residual=0.0,
            amplitudes=[np.zeros_like(v) for v in rhs],
        )

    def _flatten(vecs: list[np.ndarray]) -> np.ndarray:
        return np.concatenate([v.ravel() for v in vecs])

    def _unflatten(vec: np.ndarray) -> list[np.ndarray]:
        out = []
        offset = 0
        for s in sizes:
            out.append(vec[offset:offset + s].copy())
            offset += s
        return out

    def _precond(r: np.ndarray) -> np.ndarray:
        vecs = _unflatten(r)
        out = []
        for v, d in zip(vecs, h0_diag):
            z = np.zeros_like(v)
            mask = np.abs(d) > 1e-14
            z[mask] = v[mask] / d[mask]
            out.append(z)
        return _flatten(out)

    def _matvec(x: np.ndarray) -> np.ndarray:
        vecs = _unflatten(x)
        return _flatten(sigma_op(vecs))

    b = -_flatten(rhs)
    x = _precond(b)  # initial guess

    r = b - _matvec(x)
    z = _precond(r)
    p = z.copy()
    rz = float(np.dot(r, z))

    for it in range(maxiter):
        ap = _matvec(p)
        pap = float(np.dot(p, ap))
        if abs(pap) < 1e-30:
            break
        alpha = rz / pap
        x += alpha * p
        r -= alpha * ap

        res_norm = float(np.linalg.norm(r))
        if verbose >= 2:
            print(f"  PCG iter {it + 1}: |r| = {res_norm:.2e}")
        if res_norm < tol:
            return PCGResult(
                converged=True, niter=it + 1, residual=res_norm,
                amplitudes=_unflatten(x),
                info={"method": "pcg"},
            )

        z = _precond(r)
        rz_new = float(np.dot(r, z))
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return PCGResult(
        converged=False, niter=maxiter, residual=float(np.linalg.norm(r)),
        amplitudes=_unflatten(x),
        info={"method": "pcg"},
    )
