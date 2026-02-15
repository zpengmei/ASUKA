from __future__ import annotations

"""Dense Z-vector solver (debug/reference).

This module provides a dense linear solver for the MCSCF/CASSCF super-Hessian:

  - Explicitly builds the dense matrix A by applying the Hessian matvec n times.
  - Solves A x = b via an SVD pseudo-inverse, robust to SA-CASSCF rank deficiency.

It is intended for *small* systems and debugging. For production use, prefer the
iterative solvers in :mod:`asuka.mcscf.zvector`.
"""

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .zvector import _project_sa_ci_components


def _build_dense_matrix(mv: Callable[[np.ndarray], np.ndarray], n: int) -> np.ndarray:
    """Build dense matrix A such that A @ x == mv(x).

    Costs O(n) matvecs and O(n^2) memory; only for small n.

    Parameters
    ----------
    mv : Callable[[np.ndarray], np.ndarray]
        Matrix-vector product function.
    n : int
        Dimension of the vector space.

    Returns
    -------
    np.ndarray
        Dense matrix A of shape (n,n).
    """

    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    A = np.empty((n, n), dtype=np.float64)
    e = np.zeros((n,), dtype=np.float64)
    for j in range(n):
        e.fill(0.0)
        e[j] = 1.0
        A[:, j] = np.asarray(mv(e), dtype=np.float64).ravel()
    return np.asarray(A, dtype=np.float64)


@dataclass(frozen=True)
class DenseSVDLinearSolver:
    """Dense linear solver using an SVD pseudo-inverse.

    Solves x = A^+ b (Moore–Penrose pseudo-inverse). For SA-CASSCF, the operator
    is singular in the root-span gauge directions; SVD provides a robust solve.

    Attributes
    ----------
    A : np.ndarray
        The dense matrix (n,n).
    U : np.ndarray
        Left singular vectors.
    s : np.ndarray
        Singular values.
    Vh : np.ndarray
        Right singular vectors (transposed).
    s_inv : np.ndarray
        Inverse singular values (masked).
    rcond : float
        Condition number threshold.
    is_sa : bool
        Whether this is for state-averaged CASSCF.
    ci_ref_list : list[np.ndarray] | None
        Reference CI vectors.
    sa_gram_inv : np.ndarray | None
        Inverse Gram matrix for SA projection.
    n_orb : int
        Number of orbital parameters.
    ci_unflatten : Callable[[np.ndarray], Any]
        Unflatten function for CI vectors.
    """

    A: np.ndarray
    U: np.ndarray
    s: np.ndarray
    Vh: np.ndarray
    s_inv: np.ndarray
    rcond: float

    # SA projection metadata (mirrors MCSCFHessianOp)
    is_sa: bool
    ci_ref_list: list[np.ndarray] | None
    sa_gram_inv: np.ndarray | None
    n_orb: int
    ci_unflatten: Callable[[np.ndarray], Any]

    @classmethod
    def from_hessian_op(cls, hess_op: Any, *, rcond: float = 1e-12) -> "DenseSVDLinearSolver":
        """Create a dense solver from a Hessian operator.

        Parameters
        ----------
        hess_op : Any
            Hessian operator object (must have `mv`, `n_tot`, etc.).
        rcond : float, optional
            SVD cutoff ratio (default 1e-12).

        Returns
        -------
        DenseSVDLinearSolver
            Initialized solver.
        """
        import scipy.linalg  # local import (debug-only path)

        n = int(getattr(hess_op, "n_tot"))
        if n <= 0:
            raise ValueError("hess_op.n_tot must be positive")

        A = _build_dense_matrix(getattr(hess_op, "mv"), n)

        U, s, Vh = scipy.linalg.svd(A, full_matrices=False, lapack_driver="gesdd")
        s = np.asarray(s, dtype=np.float64).ravel()

        if s.size:
            cut = float(rcond) * float(s[0])
            mask = s > cut
            s_inv = np.zeros_like(s)
            s_inv[mask] = 1.0 / s[mask]
        else:  # pragma: no cover
            s_inv = s

        return cls(
            A=np.asarray(A, dtype=np.float64),
            U=np.asarray(U, dtype=np.float64),
            s=s,
            Vh=np.asarray(Vh, dtype=np.float64),
            s_inv=np.asarray(s_inv, dtype=np.float64),
            rcond=float(rcond),
            is_sa=bool(getattr(hess_op, "is_sa", False)),
            ci_ref_list=getattr(hess_op, "ci_ref_list", None),
            sa_gram_inv=getattr(hess_op, "sa_gram_inv", None),
            n_orb=int(getattr(hess_op, "n_orb")),
            ci_unflatten=getattr(hess_op, "ci_unflatten"),
        )

    def solve_vec(self, b: np.ndarray) -> np.ndarray:
        """Solve A x = b (pseudo-inverse).

        Parameters
        ----------
        b : np.ndarray
            RHS vector.

        Returns
        -------
        np.ndarray
            Solution vector x.
        """

        b = np.asarray(b, dtype=np.float64).ravel()
        y = self.U.T @ b
        y = self.s_inv * y
        x = self.Vh.T @ y
        return np.asarray(x, dtype=np.float64).ravel()

    def solve_many(self, B: np.ndarray) -> np.ndarray:
        """Solve A X = B for multiple RHS columns (B shape (n, m)).

        Parameters
        ----------
        B : np.ndarray
            RHS matrix (n,m).

        Returns
        -------
        np.ndarray
            Solution matrix X (n,m).
        """

        B = np.asarray(B, dtype=np.float64)
        if B.ndim != 2 or int(B.shape[0]) != int(self.A.shape[0]):
            raise ValueError("B must have shape (n, m) with n matching A")
        Y = self.U.T @ B
        Y = self.s_inv[:, None] * Y
        X = self.Vh.T @ Y
        return np.asarray(X, dtype=np.float64)

    def pack_rhs(self, *, rhs_orb: np.ndarray, rhs_ci_list: Sequence[np.ndarray]) -> np.ndarray:
        """Pack RHS as (orbital + flattened CI), projecting SA gauge components.

        Parameters
        ----------
        rhs_orb : np.ndarray
            Orbital RHS part.
        rhs_ci_list : Sequence[np.ndarray]
            List of CI RHS vectors.

        Returns
        -------
        np.ndarray
            Packed RHS vector.
        """

        rhs_orb_flat = np.asarray(rhs_orb, dtype=np.float64).ravel()
        if int(rhs_orb_flat.size) != int(self.n_orb):
            raise ValueError("rhs_orb size mismatch")

        rhs_ci_flat = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in rhs_ci_list])
        if self.is_sa:
            rhs_ci_list2 = self.ci_unflatten(rhs_ci_flat)
            if not isinstance(rhs_ci_list2, list) or self.ci_ref_list is None:
                raise RuntimeError("expected list CI structure for SA-CASSCF RHS projection")
            rhs_ci_list2 = _project_sa_ci_components(self.ci_ref_list, rhs_ci_list2, gram_inv=self.sa_gram_inv)
            rhs_ci_flat = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in rhs_ci_list2])

        return np.concatenate([rhs_orb_flat, rhs_ci_flat])

    def solve_rhs(self, *, rhs_orb: np.ndarray, rhs_ci_list: Sequence[np.ndarray]) -> np.ndarray:
        """Solve the ASUKA Z-vector convention: A z = -rhs. Returns packed z.

        Parameters
        ----------
        rhs_orb : np.ndarray
            Orbital RHS part.
        rhs_ci_list : Sequence[np.ndarray]
            List of CI RHS vectors.

        Returns
        -------
        np.ndarray
            Packed solution vector z.
        """

        rhs = self.pack_rhs(rhs_orb=rhs_orb, rhs_ci_list=rhs_ci_list)
        return self.solve_vec(-rhs)


__all__ = ["DenseSVDLinearSolver"]
