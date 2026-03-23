"""Augmented Hessian solver for second-order orbital optimization.

Maintains a subspace of trial/sigma vectors, builds the augmented matrix,
and iteratively refines the solution via Davidson-like micro-iterations.

Based on the AugHess implementation in the BAGEL package
(https://github.com/qsimulate-open/bagel, GPLv3+).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from asuka.mcscf.rotfile import RotFile


class AugHess:
    """Augmented Hessian eigenvector solver (Krylov-based eigenvector solver)."""

    def __init__(self, max_size: int, grad: RotFile, *, maxstepsize: float = 0.3):
        self._max = int(max_size)
        self._size = 0
        self._grad = grad

        n = self._max
        self._mat = np.zeros((n + 1, n + 1), dtype=np.float64)
        self._prod = np.zeros(n, dtype=np.float64)
        self._vec = np.zeros(n, dtype=np.float64)
        self._eig = np.zeros(n, dtype=np.float64)

        self._c: list[RotFile] = []
        self._sigma: list[RotFile] = []

        self._maxstepsize = float(maxstepsize)

    # ------------------------------------------------------------------ private
    def _compute_lambda(self, mat1: np.ndarray, mat2: np.ndarray) -> tuple[float, float]:
        """Trust-region lambda search.  Translation of ``compute_lambda_`` in aughess.h."""
        nlast = int(mat1.shape[0])
        assert nlast > 1
        lambda_test = 1.0
        lambda_lasttest = 0.0
        stepsize_lasttest = 0.0
        stepsize = 0.0
        iok = 0

        for i in range(10):
            scr = mat1 + mat2 * (1.0 / lambda_test)
            v, vecs = np.linalg.eigh(scr)
            # numpy eigh returns sorted ascending -- iterate to find ivec
            ivec = -1
            for j in range(nlast):
                if abs(vecs[nlast - 1, j]) <= 1.1 and abs(vecs[nlast - 1, j]) > 0.1:
                    ivec = j
                    break
            if ivec < 0:
                raise RuntimeError("logical error in AugHess._compute_lambda")

            # Build step vector norm
            coeffs = vecs[: nlast - 1, ivec] / vecs[nlast - 1, ivec]
            x = self._c[0].clone()
            for ii, ci in enumerate(self._c):
                x.ax_plus_y(coeffs[ii], ci)
            stepsize = x.norm() / abs(lambda_test)

            if i == 0:
                if stepsize <= self._maxstepsize:
                    break
                lambda_lasttest = lambda_test
                lambda_test = stepsize / self._maxstepsize
            else:
                if abs(stepsize - self._maxstepsize) / self._maxstepsize < 0.01:
                    break
                if stepsize > self._maxstepsize:
                    lambda_lasttest = lambda_test
                    lambda_test *= stepsize / self._maxstepsize
                else:
                    if iok > 2:
                        break
                    iok += 1
                    d1 = self._maxstepsize - stepsize
                    d2 = stepsize_lasttest - self._maxstepsize
                    if d1 == 0.0 or d1 == -d2:
                        break
                    lambda_lasttest_save = lambda_lasttest
                    lambda_lasttest = lambda_test
                    lambda_test = (d1 / (d1 + d2)) * lambda_lasttest_save + (d2 / (d1 + d2)) * lambda_test

            if lambda_test < 1.0:
                lambda_test = 1.0
            stepsize_lasttest = stepsize

        return (lambda_test, stepsize)

    # ------------------------------------------------------------------ update
    def _update(self, c: RotFile, s: RotFile) -> None:
        """Register a new trial vector *c* and its sigma *s* in the subspace."""
        if self._size + 1 == self._max:
            raise RuntimeError("max size reached in AugHess")
        assert abs(c.norm() - 1.0) < 1.0e-8, f"trial vector not normalized: norm={c.norm()}"

        self._c.append(c.copy())
        self._sigma.append(s.copy())
        self._size += 1

        # Update mat and prod
        k = self._size - 1
        for idx in range(self._size):
            val = 0.5 * (s.dot(self._c[idx]) + c.dot(self._sigma[idx]))
            self._mat[k, idx] = val
            self._mat[idx, k] = val
        self._prod[k] = self._grad.dot(c)

    # --------------------------------------------------------- compute_residual
    def compute_residual(
        self, c: RotFile, s: RotFile
    ) -> tuple[RotFile, float, float, float]:
        """Add trial ``(c, s)`` and compute the residual.

        Returns ``(residual, lambda, epsilon, stepsize)`` exactly matching
        the augmented Hessian eigenvalue problem.
        """
        self._update(c, s)

        n = self._size

        # scr1 = augmented gradient coupling
        scr1 = np.zeros((n + 1, n + 1), dtype=np.float64)
        for i in range(n):
            scr1[n, i] = self._prod[i]
            scr1[i, n] = self._prod[i]

        # scr2 = Hessian overlap subspace
        scr2 = self._mat[: n + 1, : n + 1].copy()

        lam, stepsize = self._compute_lambda(scr1, scr2)

        # Full eigensystem of augmented matrix with lambda
        scr = scr1 + scr2 * (1.0 / lam)
        eig_vals, scr_vecs = np.linalg.eigh(scr)

        # Find the best eigenvector
        ivec = -1
        for i in range(n + 1):
            if abs(scr_vecs[n, i]) <= 1.1 and abs(scr_vecs[n, i]) > 0.1:
                ivec = i
                break
        if ivec < 0:
            raise RuntimeError("logical error in AugHess.compute_residual")

        epsilon = float(eig_vals[ivec])

        # Build solution coefficients
        for i in range(n):
            self._vec[i] = scr_vecs[i, ivec] / (lam * scr_vecs[n, ivec])

        # Build residual: residual = grad + sum_i vec[i] * (sigma[i] - lam*eps*c[i])
        out = self._grad.copy()
        for idx in range(n):
            out.ax_plus_y(self._vec[idx], self._sigma[idx])
            out.ax_plus_y(-self._vec[idx] * lam * epsilon, self._c[idx])

        return (out, float(lam), epsilon, float(stepsize))

    # -------------------------------------------------------------- civec
    def civec(self) -> RotFile:
        """Return the current solution vector as a linear combination of trial vectors."""
        out = self._c[0].clone()
        for idx in range(self._size):
            out.ax_plus_y(self._vec[idx], self._c[idx])
        return out

    # -------------------------------------------------------------- orthog
    def orthog(self, cc: RotFile) -> float:
        """Orthogonalize *cc* against the stored trial vectors; return post-norm."""
        return cc.orthog(self._c)
