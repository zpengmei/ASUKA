from __future__ import annotations

"""Cartesian geometry optimization with energy+gradient callbacks.

This optimizer is deliberately lightweight and method-agnostic. It can be used
with:

- ASUKA gradient drivers (CASSCF, CASPT2, MRCI, SOC-SI, ...) by providing a
  wrapper ``energy_grad(coords_bohr)->(E, grad)``.

The current implementation focuses on robust *minimum* optimization using an
L-BFGS direction and a simple Armijo backtracking line search.

Optional feature
----------------
One or more scalar internal-coordinate constraints can be enforced via
projection (intended for relaxed scans). If constraints are provided, the
optimizer:

- projects geometries back onto the constraint manifold after each trial step
- projects gradients and step directions into the manifold tangent space
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .constraints import InternalCoordinateConstraint


EnergyGradFn = Callable[[np.ndarray], tuple[float, np.ndarray]]


def _flatten_coords(coords_bohr: np.ndarray) -> np.ndarray:
    c = np.asarray(coords_bohr, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError("coords must have shape (natm,3)")
    return np.asarray(c.reshape(-1), dtype=np.float64)


def _unflatten_coords(x: np.ndarray, natm: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape((-1,))
    if x.size != 3 * int(natm):
        raise ValueError("coordinate vector length mismatch")
    return np.asarray(x.reshape((int(natm), 3)), dtype=np.float64)


def _grad_stats(gvec: np.ndarray) -> tuple[float, float]:
    g = np.asarray(gvec, dtype=np.float64).reshape((-1,))
    gmax = float(np.max(np.abs(g))) if g.size else 0.0
    grms = float(np.sqrt(np.mean(g * g))) if g.size else 0.0
    return gmax, grms


def _max_atom_step(step_vec: np.ndarray, natm: int) -> float:
    s = np.asarray(step_vec, dtype=np.float64).reshape((int(natm), 3))
    return float(np.max(np.linalg.norm(s, axis=1))) if natm > 0 else 0.0


def _solve_small(G: np.ndarray, b: np.ndarray, *, eps: float = 1e-20) -> np.ndarray:
    G = np.asarray(G, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("G must be square")
    if b.shape[0] != G.shape[0]:
        raise ValueError("b dimension mismatch")
    if int(G.shape[0]) == 0:
        return np.zeros_like(b)

    diag = np.diag(G)
    if diag.size and float(np.min(np.abs(diag))) <= float(eps):  # pragma: no cover
        return np.linalg.pinv(G) @ b

    try:
        return np.linalg.solve(G, b)
    except np.linalg.LinAlgError:  # pragma: no cover
        return np.linalg.pinv(G) @ b


def _project_vec_multi(v: np.ndarray, J: np.ndarray, *, eps: float = 1e-20) -> np.ndarray:
    """Project v to the tangent space of multiple constraints.

    For constraint Jacobian rows J (shape m x n), remove components in the span
    of J^T:
        v_t = v - J^T (J J^T)^{-1} (J v)
    """

    v = np.asarray(v, dtype=np.float64).reshape((-1,))
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2 or J.shape[1] != v.size:
        raise ValueError("J must have shape (n_constraints, ndof)")
    if int(J.shape[0]) == 0:
        return np.array(v, copy=True)

    G = J @ J.T
    rhs = J @ v
    if not np.all(np.isfinite(G)) or not np.all(np.isfinite(rhs)):  # pragma: no cover
        return np.array(v, copy=True)

    lam = _solve_small(G, rhs, eps=float(eps))
    return v - (J.T @ lam)


def _constraints_value_jacobian(
    coords_bohr: np.ndarray, constraints: Sequence[InternalCoordinateConstraint]
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords_bohr, dtype=np.float64).reshape((-1, 3))
    natm = int(coords.shape[0])
    n = 3 * natm

    q = np.zeros(int(len(constraints)), dtype=np.float64)
    J = np.zeros((int(len(constraints)), n), dtype=np.float64)
    for a, c in enumerate(constraints):
        q[a] = float(c.value(coords))
        dq = np.asarray(c.jacobian(coords), dtype=np.float64)
        if dq.shape != (natm, 3):
            raise ValueError("constraint.jacobian must return shape (natm,3)")
        J[a] = dq.reshape((n,))
    return q, J


def _project_coords_multi(
    coords_bohr: np.ndarray,
    constraints: Sequence[InternalCoordinateConstraint],
    targets: np.ndarray,
    *,
    tol: float,
    max_iter: int,
    max_corr_bohr: float | None,
    eps: float = 1e-20,
) -> np.ndarray:
    coords = np.asarray(coords_bohr, dtype=np.float64).reshape((-1, 3)).copy()
    natm = int(coords.shape[0])
    n = 3 * natm

    t = np.asarray(targets, dtype=np.float64).reshape((-1,))
    if t.size != int(len(constraints)):
        raise ValueError("constraint target length mismatch")

    for _it in range(int(max_iter)):
        q, J = _constraints_value_jacobian(coords, constraints)
        resid = q - t
        if float(np.max(np.abs(resid))) <= float(tol):
            return coords

        G = J @ J.T
        lam = _solve_small(G, resid, eps=float(eps))
        step = J.T @ lam

        if max_corr_bohr is not None:
            rms = float(np.sqrt(np.mean(step * step))) if step.size else 0.0
            if rms > float(max_corr_bohr) > 0.0:
                step = step * (float(max_corr_bohr) / rms)

        coords = (coords.reshape((n,)) - step).reshape((natm, 3))

    return coords


def _lbfgs_two_loop(
    g: np.ndarray,
    s_hist: list[np.ndarray],
    y_hist: list[np.ndarray],
    rho_hist: list[float],
) -> np.ndarray:
    """Compute the L-BFGS inverse-Hessian product H_k g (not the sign)."""

    q = np.array(g, copy=True)
    alpha: list[float] = []
    for s, y, rho in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
        a = rho * float(np.dot(s, q))
        alpha.append(a)
        q = q - a * y

    if s_hist:
        s_last = s_hist[-1]
        y_last = y_hist[-1]
        yy = float(np.dot(y_last, y_last))
        sy = float(np.dot(s_last, y_last))
        gamma = sy / yy if yy > 0.0 else 1.0
    else:
        gamma = 1.0
    r = gamma * q

    for s, y, rho, a in zip(s_hist, y_hist, rho_hist, reversed(alpha)):
        b = rho * float(np.dot(y, r))
        r = r + s * (a - b)
    return r


@dataclass(frozen=True)
class GeomOptSettings:
    """Settings for Cartesian geometry optimization."""

    max_steps: int = 100

    # Convergence thresholds (Eh/Bohr)
    gmax_tol: float = 3.0e-4
    grms_tol: float = 1.0e-4

    # Step control
    step_max_bohr: float = 0.30

    # Line search (Armijo)
    ls_c1: float = 1.0e-4
    ls_beta: float = 0.5
    ls_max_iter: int = 12

    # L-BFGS
    history_size: int = 10
    curvature_tol: float = 1.0e-10

    # Constraint projection (only used when constraints are provided)
    constraint_tol: float = 1.0e-10
    constraint_max_iter: int = 25
    constraint_max_corr_bohr: float | None = None

    # Bookkeeping
    store_trajectory: bool = False
    verbose: int = 0


@dataclass(frozen=True)
class GeomOptResult:
    converged: bool
    message: str

    n_steps: int
    n_eval: int

    coords_final_bohr: np.ndarray
    energy_final: float
    grad_final: np.ndarray

    energies: np.ndarray
    gmax: np.ndarray
    grms: np.ndarray
    max_atom_step: np.ndarray

    coords_traj_bohr: np.ndarray | None = None


def optimize_cartesian(
    energy_grad: EnergyGradFn,
    coords0_bohr: np.ndarray,
    *,
    settings: GeomOptSettings | None = None,
    constraint: InternalCoordinateConstraint | None = None,
    constraint_target: float | None = None,
    constraints: Sequence[InternalCoordinateConstraint] | None = None,
    constraint_targets: Sequence[float] | None = None,
) -> GeomOptResult:
    """Optimize a molecular geometry (minimum) in Cartesian coordinates."""

    st = settings or GeomOptSettings()

    coords0 = np.asarray(coords0_bohr, dtype=np.float64)
    natm = int(coords0.shape[0])
    if coords0.shape != (natm, 3):
        raise ValueError("coords0_bohr must have shape (natm,3)")

    if constraint is not None or constraint_target is not None:
        if constraints is not None or constraint_targets is not None:
            raise ValueError("use either constraint/constraint_target or constraints/constraint_targets (not both)")
        if (constraint is None) != (constraint_target is None):
            raise ValueError("constraint and constraint_target must be provided together")
        constraints_use: list[InternalCoordinateConstraint] = [constraint]  # type: ignore[list-item]
        targets_use = np.asarray([float(constraint_target)], dtype=np.float64)  # type: ignore[arg-type]
    elif constraints is not None or constraint_targets is not None:
        if (constraints is None) != (constraint_targets is None):
            raise ValueError("constraints and constraint_targets must be provided together")
        constraints_use = list(constraints or [])
        targets_use = np.asarray(list(constraint_targets or []), dtype=np.float64)
        if int(len(constraints_use)) == 0:
            raise ValueError("constraints must be non-empty")
        if targets_use.size != int(len(constraints_use)):
            raise ValueError("constraints and constraint_targets length mismatch")
        if not np.all(np.isfinite(targets_use)):
            raise ValueError("constraint_targets must be finite")
    else:
        constraints_use = []
        targets_use = np.zeros(0, dtype=np.float64)

    # Project initial geometry if constrained.
    if constraints_use:
        if int(len(constraints_use)) == 1:
            coords0 = constraints_use[0].project(
                coords0,
                float(targets_use[0]),
                tol=float(st.constraint_tol),
                max_iter=int(st.constraint_max_iter),
                max_corr_bohr=st.constraint_max_corr_bohr,
            )
        else:
            max_corr = st.constraint_max_corr_bohr
            if max_corr is None:
                max_corr = 0.10
            coords0 = _project_coords_multi(
                coords0,
                constraints_use,
                targets_use,
                tol=float(st.constraint_tol),
                max_iter=int(st.constraint_max_iter),
                max_corr_bohr=max_corr,
            )

    x = _flatten_coords(coords0)
    ndof = int(x.size)
    if ndof == 0:
        E0, g0 = energy_grad(coords0)
        return GeomOptResult(
            converged=True,
            message="trivial (no atoms)",
            n_steps=0,
            n_eval=1,
            coords_final_bohr=np.array(coords0, copy=True),
            energy_final=float(E0),
            grad_final=np.asarray(g0, dtype=np.float64),
            energies=np.asarray([float(E0)]),
            gmax=np.asarray([0.0]),
            grms=np.asarray([0.0]),
            max_atom_step=np.asarray([0.0]),
            coords_traj_bohr=np.array(coords0[None, :, :], copy=True) if st.store_trajectory else None,
        )

    s_hist: list[np.ndarray] = []
    y_hist: list[np.ndarray] = []
    rho_hist: list[float] = []

    energies: list[float] = []
    gmax_hist: list[float] = []
    grms_hist: list[float] = []
    step_hist: list[float] = []
    coords_traj: list[np.ndarray] = []

    def _eval(x_vec: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        coords = _unflatten_coords(x_vec, natm)

        if constraints_use:
            if int(len(constraints_use)) == 1:
                coords = constraints_use[0].project(
                    coords,
                    float(targets_use[0]),
                    tol=float(st.constraint_tol),
                    max_iter=int(st.constraint_max_iter),
                    max_corr_bohr=st.constraint_max_corr_bohr,
                )
            else:
                max_corr = st.constraint_max_corr_bohr
                if max_corr is None:
                    max_corr = 0.10
                coords = _project_coords_multi(
                    coords,
                    constraints_use,
                    targets_use,
                    tol=float(st.constraint_tol),
                    max_iter=int(st.constraint_max_iter),
                    max_corr_bohr=max_corr,
                )
            x_vec = _flatten_coords(coords)

        E, g = energy_grad(coords)
        g = np.asarray(g, dtype=np.float64)
        if g.shape != (natm, 3):
            raise ValueError("energy_grad must return grad with shape (natm,3)")
        g_raw = _flatten_coords(g)

        if not constraints_use:
            return float(E), g_raw, g_raw, x_vec, None

        _q, J = _constraints_value_jacobian(coords, constraints_use)
        g_proj = _project_vec_multi(g_raw, J)
        return float(E), g_proj, g_raw, x_vec, J

    n_eval = 0
    E, g, g_raw, x, J = _eval(x)
    n_eval += 1

    if st.store_trajectory:
        coords_traj.append(_unflatten_coords(x, natm))

    for it in range(int(st.max_steps) + 1):
        gmax, grms = _grad_stats(g)
        energies.append(float(E))
        gmax_hist.append(float(gmax))
        grms_hist.append(float(grms))
        step_hist.append(0.0 if it == 0 else float(step_hist[-1]))

        if int(st.verbose) >= 1:
            tag = "constrained" if constraints_use else "free"
            print(f"[geomopt:{tag}] iter={it:3d}  E={E:+.12f}  gmax={gmax:.3e}  grms={grms:.3e}")

        if (gmax <= float(st.gmax_tol)) and (grms <= float(st.grms_tol)):
            return GeomOptResult(
                converged=True,
                message="converged (gradient thresholds)",
                n_steps=it,
                n_eval=n_eval,
                coords_final_bohr=_unflatten_coords(x, natm),
                energy_final=float(E),
                grad_final=_unflatten_coords(g_raw, natm),
                energies=np.asarray(energies, dtype=np.float64),
                gmax=np.asarray(gmax_hist, dtype=np.float64),
                grms=np.asarray(grms_hist, dtype=np.float64),
                max_atom_step=np.asarray(step_hist, dtype=np.float64),
                coords_traj_bohr=np.asarray(coords_traj, dtype=np.float64) if st.store_trajectory else None,
            )

        if it == int(st.max_steps):
            break

        if s_hist:
            Hg = _lbfgs_two_loop(g, s_hist, y_hist, rho_hist)
            p = -Hg
        else:
            p = -g

        if J is not None:
            p = _project_vec_multi(p, J)

        slope = float(np.dot(g, p))
        if not np.isfinite(slope) or slope >= 0.0:
            p = -g
            if J is not None:
                p = _project_vec_multi(p, J)
            slope = float(np.dot(g, p))

        max_atom = _max_atom_step(p, natm)
        if max_atom > float(st.step_max_bohr):
            p = p * (float(st.step_max_bohr) / max_atom)

        alpha = 1.0
        accepted = False
        for _ls in range(int(st.ls_max_iter)):
            x_trial = x + alpha * p
            E_trial, g_trial, g_trial_raw, x_trial_proj, J_trial = _eval(x_trial)
            n_eval += 1
            if E_trial <= E + float(st.ls_c1) * alpha * slope:
                accepted = True
                break
            alpha *= float(st.ls_beta)

        if not accepted:
            alpha = float(st.ls_beta) ** float(st.ls_max_iter)
            x_trial = x + alpha * p
            E_trial, g_trial, g_trial_raw, x_trial_proj, J_trial = _eval(x_trial)
            n_eval += 1

        s = np.asarray(x_trial_proj - x, dtype=np.float64)
        y = np.asarray(g_trial - g, dtype=np.float64)
        ys = float(np.dot(y, s))
        if ys > float(st.curvature_tol):
            rho = 1.0 / ys
            s_hist.append(np.array(s, copy=True))
            y_hist.append(np.array(y, copy=True))
            rho_hist.append(float(rho))
            if len(s_hist) > int(st.history_size):
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)

        x = np.asarray(x_trial_proj, dtype=np.float64)
        E = float(E_trial)
        g = np.asarray(g_trial, dtype=np.float64)
        g_raw = np.asarray(g_trial_raw, dtype=np.float64)
        J = J_trial

        step_hist[-1] = float(_max_atom_step(s, natm))
        if st.store_trajectory:
            coords_traj.append(_unflatten_coords(x, natm))

    return GeomOptResult(
        converged=False,
        message="not converged (max_steps reached)",
        n_steps=int(st.max_steps),
        n_eval=n_eval,
        coords_final_bohr=_unflatten_coords(x, natm),
        energy_final=float(E),
        grad_final=_unflatten_coords(g_raw, natm),
        energies=np.asarray(energies, dtype=np.float64),
        gmax=np.asarray(gmax_hist, dtype=np.float64),
        grms=np.asarray(grms_hist, dtype=np.float64),
        max_atom_step=np.asarray(step_hist, dtype=np.float64),
        coords_traj_bohr=np.asarray(coords_traj, dtype=np.float64) if st.store_trajectory else None,
    )


__all__ = [
    "EnergyGradFn",
    "GeomOptSettings",
    "GeomOptResult",
    "optimize_cartesian",
]
