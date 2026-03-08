from __future__ import annotations

"""Minimum-energy conical intersection (MECI) optimization utilities.

This module implements a lightweight, method-agnostic Cartesian MECI optimizer
driven by:
  - a multiroot energy+gradient callback, and
  - a branching-plane coupling vector (h-vector) callback.

Current scope (v1)
------------------
Only the "branching-plane / NAC-based" approach is implemented. The optimizer
minimizes the average energy in the *intersection space* while enforcing
degeneracy via a quadratic gap penalty.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .optimizer import _flatten_coords, _grad_stats, _lbfgs_two_loop, _max_atom_step, _unflatten_coords


MultiRootEvalFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Any]]
HVecFn = Callable[[np.ndarray, Any, int, int], np.ndarray]


def _as_multiroot_eval_output(
    out: tuple[Any, ...],
    *,
    natm: int,
) -> tuple[np.ndarray, np.ndarray, Any | None]:
    if not isinstance(out, tuple):
        raise TypeError("multiroot_eval must return a tuple")
    if len(out) == 2:
        e_roots, grads = out
        ctx = None
    elif len(out) == 3:
        e_roots, grads, ctx = out
    else:
        raise TypeError("multiroot_eval must return (e_roots, grads) or (e_roots, grads, ctx)")

    e = np.asarray(e_roots, dtype=np.float64).reshape((-1,))
    g = np.asarray(grads, dtype=np.float64)
    if g.ndim != 3 or g.shape[1:] != (int(natm), 3):
        raise ValueError("multiroot_eval grads must have shape (nroots,natm,3)")
    if int(e.size) != int(g.shape[0]):
        raise ValueError("multiroot_eval energies/grads root dimension mismatch")
    return e, g, ctx


def _normalize(v: np.ndarray, *, eps: float) -> np.ndarray | None:
    v = np.asarray(v, dtype=np.float64).reshape((-1,))
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= float(eps):
        return None
    return np.asarray(v / n, dtype=np.float64)


def _branching_plane_basis(
    g_gap: np.ndarray,
    h_vec: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (g_hat, h_hat) as orthonormal vectors in flattened space."""

    g_hat = _normalize(g_gap, eps=float(eps))
    h = np.asarray(h_vec, dtype=np.float64).reshape((-1,))
    if g_hat is not None:
        h = h - g_hat * float(np.dot(g_hat, h))
    h_hat = _normalize(h, eps=float(eps))
    return g_hat, h_hat


def _project_is(v: np.ndarray, g_hat: np.ndarray | None, h_hat: np.ndarray | None) -> np.ndarray:
    """Project vector onto intersection space (remove branching-plane components)."""

    v = np.asarray(v, dtype=np.float64).reshape((-1,))
    out = np.array(v, copy=True)
    if g_hat is not None:
        out = out - g_hat * float(np.dot(g_hat, out))
    if h_hat is not None:
        out = out - h_hat * float(np.dot(h_hat, out))
    return out


def _remove_h_component(v: np.ndarray, h_hat: np.ndarray | None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape((-1,))
    if h_hat is None:
        return np.array(v, copy=True)
    return v - h_hat * float(np.dot(h_hat, v))


@dataclass(frozen=True)
class MECISettings:
    """Settings for Cartesian MECI optimization."""

    max_steps: int = 100

    # Convergence thresholds
    gmax_tol: float = 3.0e-4
    grms_tol: float = 1.0e-4
    gap_tol: float = 1.0e-4

    # Penalty function weight for w * gap^2 (units: Eh^-1)
    penalty_w: float = 50.0

    # Step control
    step_max_bohr: float = 0.20

    # Line search (Armijo on merit function)
    ls_c1: float = 1.0e-4
    ls_beta: float = 0.5
    ls_max_iter: int = 6

    # L-BFGS
    history_size: int = 10
    curvature_tol: float = 1.0e-10

    # Coupling vector handling
    strict_hvec: bool = True
    basis_eps: float = 1.0e-12

    # Bookkeeping
    store_trajectory: bool = False
    verbose: int = 0

    # Optional per-iteration callback:
    # fn(iter, e_avg, gap, gmax_is, grms_is, max_steps)
    callback: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class MECIResult:
    converged: bool
    message: str

    n_steps: int
    n_eval_multiroot: int
    n_eval_hvec: int

    roots: tuple[int, int]

    coords_final_bohr: np.ndarray
    e_i_final: float
    e_j_final: float
    e_avg_final: float
    gap_final: float

    grad_i_final: np.ndarray
    grad_j_final: np.ndarray

    e_avg: np.ndarray
    gap: np.ndarray
    merit: np.ndarray
    gmax_is: np.ndarray
    grms_is: np.ndarray
    max_atom_step: np.ndarray

    coords_traj_bohr: np.ndarray | None = None


def optimize_meci_cartesian(
    multiroot_eval: MultiRootEvalFn,
    coords0_bohr: np.ndarray,
    *,
    roots: tuple[int, int] = (0, 1),
    hvec: HVecFn,
    settings: MECISettings | None = None,
) -> MECIResult:
    """Optimize a MECI geometry in Cartesian coordinates.

    Parameters
    ----------
    multiroot_eval
        Callback returning either ``(e_roots, grads)`` or ``(e_roots, grads, ctx)``,
        where ``e_roots`` has shape ``(nroots,)`` and ``grads`` has shape
        ``(nroots,natm,3)``.
    coords0_bohr
        Initial geometry, shape ``(natm,3)`` in Bohr.
    roots
        Pair of root indices (i, j) defining the intersection.
    hvec
        Callback returning the h-vector (branching-plane coupling numerator)
        in Eh/Bohr, shape ``(natm,3)``.
    settings
        MECI optimization settings.
    """

    st = settings or MECISettings()

    coords0 = np.asarray(coords0_bohr, dtype=np.float64)
    natm = int(coords0.shape[0])
    if coords0.shape != (natm, 3):
        raise ValueError("coords0_bohr must have shape (natm,3)")

    i, j = (int(roots[0]), int(roots[1]))
    if i == j:
        raise ValueError("roots must be distinct")

    x = _flatten_coords(coords0)
    ndof = int(x.size)
    if ndof == 0:
        out0 = _as_multiroot_eval_output(multiroot_eval(coords0), natm=natm)
        e_roots0, g_roots0, _ctx0 = out0
        if max(i, j) >= int(e_roots0.size):
            raise IndexError("root index out of range")
        Ei = float(e_roots0[i])
        Ej = float(e_roots0[j])
        gap = float(Ej - Ei)
        e_avg = 0.5 * (Ei + Ej)
        return MECIResult(
            converged=abs(gap) <= float(st.gap_tol),
            message="trivial (no atoms)",
            n_steps=0,
            n_eval_multiroot=1,
            n_eval_hvec=0,
            roots=(i, j),
            coords_final_bohr=np.array(coords0, copy=True),
            e_i_final=Ei,
            e_j_final=Ej,
            e_avg_final=float(e_avg),
            gap_final=float(gap),
            grad_i_final=np.zeros((0, 3), dtype=np.float64),
            grad_j_final=np.zeros((0, 3), dtype=np.float64),
            e_avg=np.asarray([float(e_avg)], dtype=np.float64),
            gap=np.asarray([float(gap)], dtype=np.float64),
            merit=np.asarray([float(e_avg + float(st.penalty_w) * gap * gap)], dtype=np.float64),
            gmax_is=np.asarray([0.0], dtype=np.float64),
            grms_is=np.asarray([0.0], dtype=np.float64),
            max_atom_step=np.asarray([0.0], dtype=np.float64),
            coords_traj_bohr=np.array(coords0[None, :, :], copy=True) if st.store_trajectory else None,
        )

    s_hist: list[np.ndarray] = []
    y_hist: list[np.ndarray] = []
    rho_hist: list[float] = []

    eavg_hist: list[float] = []
    gap_hist: list[float] = []
    merit_hist: list[float] = []
    gmax_hist: list[float] = []
    grms_hist: list[float] = []
    step_hist: list[float] = []
    coords_traj: list[np.ndarray] = []

    n_eval_multiroot = 0
    n_eval_hvec = 0

    def _eval_multiroot(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, Any | None]:
        nonlocal n_eval_multiroot
        e_roots, grads, ctx = _as_multiroot_eval_output(multiroot_eval(coords), natm=natm)
        n_eval_multiroot += 1
        if max(i, j) >= int(e_roots.size):
            raise IndexError("root index out of range")
        return e_roots, grads, ctx

    def _eval_hvec(coords: np.ndarray, ctx: Any | None) -> np.ndarray:
        nonlocal n_eval_hvec
        try:
            hv = np.asarray(hvec(coords, ctx, i, j), dtype=np.float64)
        except Exception as err:
            if bool(st.strict_hvec):
                raise
            if int(st.verbose) >= 1:
                print(f"[meci] hvec failed; continuing without h-vector: {type(err).__name__}: {err}")
            hv = np.zeros((natm, 3), dtype=np.float64)
        n_eval_hvec += 1
        if hv.shape != (natm, 3):
            raise ValueError("hvec must return shape (natm,3)")
        return hv

    def _state_quantities(
        e_roots: np.ndarray, grads: np.ndarray
    ) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
        Ei = float(e_roots[i])
        Ej = float(e_roots[j])
        e_avg = 0.5 * (Ei + Ej)
        gap = Ej - Ei
        Gi = np.asarray(grads[i], dtype=np.float64)
        Gj = np.asarray(grads[j], dtype=np.float64)
        G_avg = 0.5 * (Gi + Gj)
        G_gap = (Gj - Gi)
        return Ei, Ej, float(e_avg), float(gap), G_avg, G_gap

    # Initial evaluation
    e_roots, grads, ctx = _eval_multiroot(coords0)
    hv = _eval_hvec(coords0, ctx)

    Ei, Ej, e_avg, gap, G_avg, G_gap = _state_quantities(e_roots, grads)
    g_gap_flat = _flatten_coords(G_gap)
    hv_flat = _flatten_coords(hv)
    g_hat, h_hat = _branching_plane_basis(g_gap_flat, hv_flat, eps=float(st.basis_eps))

    g_is_flat = _project_is(_flatten_coords(G_avg), g_hat, h_hat)
    g_drive = np.asarray(g_is_flat + (2.0 * float(st.penalty_w) * float(gap) * g_gap_flat), dtype=np.float64)
    grad_F_flat = np.asarray(_flatten_coords(G_avg) + (2.0 * float(st.penalty_w) * float(gap) * g_gap_flat), dtype=np.float64)
    merit = float(e_avg + float(st.penalty_w) * float(gap) * float(gap))

    if st.store_trajectory:
        coords_traj.append(np.array(coords0, copy=True))

    for it in range(int(st.max_steps) + 1):
        gmax_is, grms_is = _grad_stats(g_is_flat)

        eavg_hist.append(float(e_avg))
        gap_hist.append(float(gap))
        merit_hist.append(float(merit))
        gmax_hist.append(float(gmax_is))
        grms_hist.append(float(grms_is))
        step_hist.append(0.0 if it == 0 else float(step_hist[-1]))

        if int(st.verbose) >= 1:
            print(
                f"[meci] iter={it:3d}  Eavg={e_avg:+.12f}  gap={gap:+.3e}  "
                f"gmax_is={gmax_is:.3e}  grms_is={grms_is:.3e}"
            )

        if st.callback is not None:
            try:
                st.callback(it, float(e_avg), float(gap), float(gmax_is), float(grms_is), int(st.max_steps))
            except Exception:
                pass

        if (abs(float(gap)) <= float(st.gap_tol)) and (gmax_is <= float(st.gmax_tol)) and (grms_is <= float(st.grms_tol)):
            coords_final = _unflatten_coords(x, natm)
            return MECIResult(
                converged=True,
                message="converged (gap + projected gradient thresholds)",
                n_steps=it,
                n_eval_multiroot=n_eval_multiroot,
                n_eval_hvec=n_eval_hvec,
                roots=(i, j),
                coords_final_bohr=np.asarray(coords_final, dtype=np.float64),
                e_i_final=float(Ei),
                e_j_final=float(Ej),
                e_avg_final=float(e_avg),
                gap_final=float(gap),
                grad_i_final=np.asarray(grads[i], dtype=np.float64),
                grad_j_final=np.asarray(grads[j], dtype=np.float64),
                e_avg=np.asarray(eavg_hist, dtype=np.float64),
                gap=np.asarray(gap_hist, dtype=np.float64),
                merit=np.asarray(merit_hist, dtype=np.float64),
                gmax_is=np.asarray(gmax_hist, dtype=np.float64),
                grms_is=np.asarray(grms_hist, dtype=np.float64),
                max_atom_step=np.asarray(step_hist, dtype=np.float64),
                coords_traj_bohr=np.asarray(coords_traj, dtype=np.float64) if st.store_trajectory else None,
            )

        if it == int(st.max_steps):
            break

        # L-BFGS direction on the driving gradient.
        if s_hist:
            Hg = _lbfgs_two_loop(g_drive, s_hist, y_hist, rho_hist)
            p = -np.asarray(Hg, dtype=np.float64)
        else:
            p = -np.asarray(g_drive, dtype=np.float64)

        # Enforce "no h-vector" motion in the branching plane.
        p = _remove_h_component(p, h_hat)

        slope = float(np.dot(grad_F_flat, p))
        if (not np.isfinite(slope)) or slope >= 0.0:
            p = -np.asarray(grad_F_flat, dtype=np.float64)
            p = _remove_h_component(p, h_hat)
            slope = float(np.dot(grad_F_flat, p))

        max_atom = _max_atom_step(p, natm)
        if max_atom > float(st.step_max_bohr) > 0.0:
            p = p * (float(st.step_max_bohr) / max_atom)

        # Armijo line search on merit F = Eavg + w*gap^2 (uses only multiroot_eval).
        alpha = 1.0
        accepted = False
        x_trial = None
        coords_trial = None
        e_roots_trial = None
        grads_trial = None
        ctx_trial = None
        Ei_trial = Ej_trial = e_avg_trial = gap_trial = merit_trial = 0.0

        for _ls in range(int(st.ls_max_iter)):
            x_t = x + alpha * p
            coords_t = _unflatten_coords(x_t, natm)
            e_t, g_t, ctx_t = _eval_multiroot(coords_t)
            Ei_t, Ej_t, eavg_t, gap_t, _Gavg_t, _Ggap_t = _state_quantities(e_t, g_t)
            merit_t = float(eavg_t + float(st.penalty_w) * float(gap_t) * float(gap_t))
            if merit_t <= float(merit) + float(st.ls_c1) * float(alpha) * float(slope):
                accepted = True
                x_trial = x_t
                coords_trial = coords_t
                e_roots_trial = e_t
                grads_trial = g_t
                ctx_trial = ctx_t
                Ei_trial, Ej_trial, e_avg_trial, gap_trial, merit_trial = Ei_t, Ej_t, eavg_t, gap_t, merit_t
                break
            alpha *= float(st.ls_beta)

        if not accepted:
            alpha = float(st.ls_beta) ** float(st.ls_max_iter)
            x_t = x + alpha * p
            coords_t = _unflatten_coords(x_t, natm)
            e_t, g_t, ctx_t = _eval_multiroot(coords_t)
            Ei_t, Ej_t, eavg_t, gap_t, _Gavg_t, _Ggap_t = _state_quantities(e_t, g_t)
            merit_t = float(eavg_t + float(st.penalty_w) * float(gap_t) * float(gap_t))
            x_trial = x_t
            coords_trial = coords_t
            e_roots_trial = e_t
            grads_trial = g_t
            ctx_trial = ctx_t
            Ei_trial, Ej_trial, e_avg_trial, gap_trial, merit_trial = Ei_t, Ej_t, eavg_t, gap_t, merit_t

        assert x_trial is not None
        assert coords_trial is not None
        assert e_roots_trial is not None
        assert grads_trial is not None

        # Compute h-vector at accepted geometry (expensive) and update gradients/basis.
        hv_next = _eval_hvec(coords_trial, ctx_trial)
        Ei_next, Ej_next, e_avg_next, gap_next, G_avg_next, G_gap_next = _state_quantities(e_roots_trial, grads_trial)
        g_gap_next = _flatten_coords(G_gap_next)
        hv_next_flat = _flatten_coords(hv_next)
        g_hat_next, h_hat_next = _branching_plane_basis(g_gap_next, hv_next_flat, eps=float(st.basis_eps))

        g_is_next = _project_is(_flatten_coords(G_avg_next), g_hat_next, h_hat_next)
        g_drive_next = np.asarray(
            g_is_next + (2.0 * float(st.penalty_w) * float(gap_next) * g_gap_next),
            dtype=np.float64,
        )
        grad_F_next = np.asarray(
            _flatten_coords(G_avg_next) + (2.0 * float(st.penalty_w) * float(gap_next) * g_gap_next),
            dtype=np.float64,
        )

        s = np.asarray(x_trial - x, dtype=np.float64)
        y = np.asarray(g_drive_next - g_drive, dtype=np.float64)
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

        # Advance state.
        x = np.asarray(x_trial, dtype=np.float64)
        Ei = float(Ei_next)
        Ej = float(Ej_next)
        e_avg = float(e_avg_next)
        gap = float(gap_next)
        merit = float(merit_trial)
        e_roots = np.asarray(e_roots_trial, dtype=np.float64)
        grads = np.asarray(grads_trial, dtype=np.float64)
        ctx = ctx_trial
        hv = hv_next
        g_hat, h_hat = g_hat_next, h_hat_next
        g_is_flat = np.asarray(g_is_next, dtype=np.float64)
        g_drive = np.asarray(g_drive_next, dtype=np.float64)
        grad_F_flat = np.asarray(grad_F_next, dtype=np.float64)

        step_hist[-1] = float(_max_atom_step(s, natm))
        if st.store_trajectory:
            coords_traj.append(_unflatten_coords(x, natm))

    coords_final = _unflatten_coords(x, natm)
    return MECIResult(
        converged=False,
        message="not converged (max_steps reached)",
        n_steps=int(st.max_steps),
        n_eval_multiroot=n_eval_multiroot,
        n_eval_hvec=n_eval_hvec,
        roots=(i, j),
        coords_final_bohr=np.asarray(coords_final, dtype=np.float64),
        e_i_final=float(Ei),
        e_j_final=float(Ej),
        e_avg_final=float(e_avg),
        gap_final=float(gap),
        grad_i_final=np.asarray(grads[i], dtype=np.float64),
        grad_j_final=np.asarray(grads[j], dtype=np.float64),
        e_avg=np.asarray(eavg_hist, dtype=np.float64),
        gap=np.asarray(gap_hist, dtype=np.float64),
        merit=np.asarray(merit_hist, dtype=np.float64),
        gmax_is=np.asarray(gmax_hist, dtype=np.float64),
        grms_is=np.asarray(grms_hist, dtype=np.float64),
        max_atom_step=np.asarray(step_hist, dtype=np.float64),
        coords_traj_bohr=np.asarray(coords_traj, dtype=np.float64) if st.store_trajectory else None,
    )


__all__ = ["HVecFn", "MECIResult", "MECISettings", "MultiRootEvalFn", "optimize_meci_cartesian"]
