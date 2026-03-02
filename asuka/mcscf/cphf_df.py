from __future__ import annotations

"""RHF CPHF/Z-vector solver over DF-JK.

This module provides a minimal CPHF solver for restricted HF orbital-response problems.
It solves the linear equation:
    (eps_a - eps_i) X_ai + V[X]_ai = RHS_ai
where `V[X]` is the induced DF-HF potential projected to the (virtual,occupied)
block, matching the `scf.cphf.solve` conventions used by `pyscf.grad.casci`.

Design notes
------------
- No PySCF import at module scope.
- Pure NumPy implementation (works with the current DF nuclear-gradient codepath
  which converts arrays to NumPy).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.hf import df_scf as _df_scf


def _as_f64(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


@dataclass(frozen=True)
class RHFCPHFResult:
    """Result of an RHF CPHF calculation.

    Attributes
    ----------
    converged : bool
        Whether the solver converged.
    niter : int
        Number of iterations performed.
    residual_norm : float
        Norm of the final residual vector.
    x_vo : np.ndarray
        Solution vector, shape (nvir, nocc).
    info : dict[str, Any]
        Additional solver information/statistics.
    """
    converged: bool
    niter: int
    residual_norm: float
    x_vo: np.ndarray  # (nvir,nocc)
    info: dict[str, Any]


class _DIISVec:
    """Simple DIIS helper for vector extrapolation.

    Attributes
    ----------
    max_vec : int
        Maximum number of vectors to store.
    """
    def __init__(self, max_vec: int = 8):
        self.max_vec = int(max_vec)
        self._x: list[np.ndarray] = []
        self._e: list[np.ndarray] = []

    def push(self, x: np.ndarray, e: np.ndarray):
        self._x.append(np.asarray(x, dtype=np.float64).ravel())
        self._e.append(np.asarray(e, dtype=np.float64).ravel())
        if len(self._x) > self.max_vec:
            self._x.pop(0)
            self._e.pop(0)

    def extrapolate(self) -> np.ndarray:
        n = len(self._x)
        if n < 2:
            return self._x[-1]

        E = np.stack(self._e, axis=0)  # (n, m)
        G = E @ E.T  # (n, n)

        B = np.empty((n + 1, n + 1), dtype=np.float64)
        B[:n, :n] = G
        B[:n, n] = -1.0
        B[n, :n] = -1.0
        B[n, n] = 0.0

        rhs = np.zeros((n + 1,), dtype=np.float64)
        rhs[n] = -1.0
        coeff = np.linalg.solve(B, rhs)[:n]  # (n,)

        X = np.stack(self._x, axis=0)  # (n, m)
        return coeff @ X


def solve_rhf_cphf_df(
    B_ao: Any,
    *,
    orbo: Any,
    orbv: Any,
    eps_occ: Any,
    eps_vir: Any,
    rhs_vo: Any,
    max_cycle: int = 30,
    tol: float = 1e-10,
    denom_floor: float = 1e-12,
    level_shift: float = 0.0,
    diis: bool = True,
    diis_start_cycle: int = 1,
    diis_space: int = 8,
    damping: float = 0.0,
) -> RHFCPHFResult:
    """Solve the RHF CPHF linear system for the (virtual,occupied) block.

    Parameters
    ----------
    B_ao : Any
        Whitened DF factors, shape (nao, nao, naux).
    orbo : Any
        Occupied MO coefficient block in AO basis.
    orbv : Any
        Virtual MO coefficient block in AO basis.
    eps_occ : Any
        Orbital energies for occupied subspace.
    eps_vir : Any
        Orbital energies for virtual subspace.
    rhs_vo : Any
        Right-hand side vector/matrix, shape (nvir, nocc).
    max_cycle : int, optional
        Maximum number of iterations. Defaults to 30.
    tol : float, optional
        Convergence tolerance for residual norm. Defaults to 1e-10.
    denom_floor : float, optional
        Minimum value for the energy denominator. Defaults to 1e-12.
    level_shift : float, optional
        Level shift parameter. Defaults to 0.0.
    diis : bool, optional
        Whether to use DIIS extrapolation. Defaults to True.
    diis_start_cycle : int, optional
        Cycle to start DIIS. Defaults to 1.
    diis_space : int, optional
        DIIS subspace size. Defaults to 8.
    damping : float, optional
        Damping factor (0.0 means no damping). Defaults to 0.0.

    Returns
    -------
    RHFCPHFResult
        The result of the CPHF calculation.
    """

    B_ao = _as_f64(B_ao)
    orbo = _as_f64(orbo)
    orbv = _as_f64(orbv)
    eps_occ = _as_f64(eps_occ).ravel()
    eps_vir = _as_f64(eps_vir).ravel()
    rhs = _as_f64(rhs_vo)

    if orbo.ndim != 2 or orbv.ndim != 2:
        raise ValueError("orbo/orbv must be 2D")
    nao0 = int(orbo.shape[0])
    if int(orbv.shape[0]) != nao0:
        raise ValueError("orbo/orbv nao mismatch")
    nocc = int(orbo.shape[1])
    nvir = int(orbv.shape[1])
    if rhs.shape != (nvir, nocc):
        raise ValueError(f"rhs_vo must have shape ({nvir},{nocc}), got {tuple(rhs.shape)}")
    if eps_occ.shape != (nocc,) or eps_vir.shape != (nvir,):
        raise ValueError("eps_occ/eps_vir shape mismatch with orbo/orbv")

    denom = eps_vir[:, None] + float(level_shift) - eps_occ[None, :]
    denom = np.where(np.abs(denom) < float(denom_floor), float(denom_floor), denom)

    def fvind(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape((nvir, nocc))
        dm = orbv @ x @ orbo.T
        dm = dm + dm.T
        J, K = _df_scf._df_JK(B_ao, dm, want_J=True, want_K=True)  # noqa: SLF001
        v_ao = J - 0.5 * K
        v_vo = orbv.T @ v_ao @ orbo
        return 2.0 * np.asarray(v_vo, dtype=np.float64)

    # Match PySCF's `scf.cphf.solve_nos1` which solves:
    #     denom * x + fvind(x) = -rhs
    # with initial guess -rhs/denom (fvind assumed 0 at the start).
    x = -rhs / denom
    diis_obj = _DIISVec(max_vec=int(diis_space)) if bool(diis) else None

    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        return RHFCPHFResult(converged=True, niter=0, residual_norm=0.0, x_vo=np.zeros_like(rhs), info={"rhs_norm": 0.0})

    converged = False
    res_norm = float("inf")
    for it in range(1, int(max_cycle) + 1):
        v = fvind(x)
        r = (-rhs - v) - (denom * x)
        res_norm = float(np.linalg.norm(r))
        if res_norm <= float(tol) * rhs_norm:
            converged = True
            break

        dx = r / denom
        x_new = x + dx
        damp = float(damping)
        if damp:
            x_new = (1.0 - damp) * x_new + damp * x

        if diis_obj is not None:
            diis_obj.push(x_new, r)
            if it >= int(diis_start_cycle) and len(diis_obj._x) >= 2:
                x = diis_obj.extrapolate().reshape((nvir, nocc))
            else:
                x = x_new
        else:
            x = x_new

    info = {
        "rhs_norm": float(rhs_norm),
        "residual_rel": float(res_norm / rhs_norm) if rhs_norm else float("nan"),
        "diis": bool(diis),
        "diis_space": int(diis_space),
        "max_cycle": int(max_cycle),
        "level_shift": float(level_shift),
    }
    return RHFCPHFResult(converged=bool(converged), niter=int(it if rhs_norm else 0), residual_norm=float(res_norm), x_vo=np.asarray(x, dtype=np.float64), info=info)
