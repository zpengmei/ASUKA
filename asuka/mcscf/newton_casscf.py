from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import time
from typing import Any, Callable, Iterator, Sequence

import numpy as np

try:
    import cupy as _cp  # type: ignore
except Exception:
    _cp = None  # type: ignore


def _get_xp(*arrays: Any) -> tuple[Any, bool]:
    """Return (xp, is_gpu) based on array types.

    Parameters
    ----------
    *arrays : Any
        Arrays to inspect.

    Returns
    -------
    xp : module
        The array module (numpy or cupy).
    is_gpu : bool
        Whether the arrays are on GPU.
    """
    if _cp is not None:
        for a in arrays:
            if isinstance(a, _cp.ndarray):
                return _cp, True
    return np, False


def _to_np_f64(a: Any) -> np.ndarray:
    """Convert array (numpy or cupy) to numpy float64."""
    if hasattr(a, "get"):  # CuPy ndarray
        a = a.get()
    return np.asarray(a, dtype=np.float64)


def _to_xp_f64(a: Any, xp: Any = None) -> Any:
    """Convert array to float64, preserving device (GPU or CPU).

    Parameters
    ----------
    a : Any
        The input array.
    xp : Any, optional
        The array module to use. If None, it is inferred from `a`.

    Returns
    -------
    Any
        The converted array.
    """
    if xp is None:
        xp, _ = _get_xp(a)
    return xp.asarray(a, dtype=xp.float64)


class _SimpleLogger:
    """Tiny logger adapter with the subset used by this module.

    Attributes
    ----------
    verbose : int
        Verbosity level.
    """

    QUIET = 0
    WARN = 2
    INFO = 4
    DEBUG = 5
    DEBUG1 = 6

    def __init__(self, verbose: int = QUIET):
        self.verbose = int(verbose)

    @staticmethod
    def _fmt(msg: str, args: tuple[Any, ...]) -> str:
        if not args:
            return str(msg)
        try:
            return str(msg) % args
        except Exception:
            return f"{msg} {' '.join(str(x) for x in args)}"

    def debug(self, msg: str, *args: Any) -> None:
        if self.verbose >= self.DEBUG:
            print(self._fmt(msg, args))

    def debug1(self, msg: str, *args: Any) -> None:
        if self.verbose >= self.DEBUG1:
            print(self._fmt(msg, args))

    def info(self, msg: str, *args: Any) -> None:
        if self.verbose >= self.INFO:
            print(self._fmt(msg, args))

    def warn(self, msg: str, *args: Any) -> None:
        if self.verbose >= self.WARN:
            print(self._fmt(msg, args))

    def timer(self, label: str, t0_cpu: float, t0_wall: float) -> tuple[float, float]:
        t1 = (time.process_time(), time.perf_counter())
        self.debug("%s: CPU %.2f sec, wall %.2f sec", label, t1[0] - t0_cpu, t1[1] - t0_wall)
        return t1


def _new_logger(obj: Any | None = None, verbose: Any | None = None) -> _SimpleLogger:
    if isinstance(verbose, _SimpleLogger):
        return verbose
    if verbose is None:
        if obj is not None:
            verbose = getattr(obj, "verbose", _SimpleLogger.QUIET)
        else:
            verbose = _SimpleLogger.QUIET
    return _SimpleLogger(int(verbose))


def _safe_eigh(h: np.ndarray, s: np.ndarray, lindep: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seig, t = np.linalg.eigh(np.asarray(s, dtype=np.float64))
    mask = seig >= float(lindep)
    t = t[:, mask]
    if t.size == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros_like(t), seig
    t = t * (1.0 / np.sqrt(seig[mask]))
    heff = t.conj().T @ np.asarray(h, dtype=np.float64) @ t
    w, v = np.linalg.eigh(heff)
    return w, t @ v, seig


def _dgemv(v: np.ndarray, m: Sequence[np.ndarray]) -> np.ndarray:
    out = np.asarray(v[0], dtype=np.float64) * np.asarray(m[0], dtype=np.float64)
    for i, vi in enumerate(np.asarray(v[1:], dtype=np.float64)):
        out += float(vi) * np.asarray(m[i + 1], dtype=np.float64)
    return out


def _regular_step(
    heff: np.ndarray,
    ovlp: np.ndarray,
    xs: Sequence[np.ndarray],
    ax: Sequence[np.ndarray] | None,
    lindep: float,
    log: _SimpleLogger,
    *,
    v_prev: np.ndarray | None = None,
    root_v0_min: float = 0.1,
    root_homing: bool = False,
    root_pred_decrease: bool = False,
    root_pred_decrease_tol_rel: float = 1e-3,
    trust_maxabs_orb: float | None = None,
    trust_maxabs_ci: float | None = None,
    ngorb: int | None = None,
    mu_orb: float = 0.0,
    mu_ci: float = 0.0,
    ovlp_orb: np.ndarray | None = None,
    ovlp_ci: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray, int, np.ndarray]:
    w, v, seig = _safe_eigh(heff, ovlp, lindep)
    if w.size == 0 or v.shape[1] == 0:
        return np.zeros_like(xs[0]), 0.0, np.zeros((0,), dtype=np.float64), 0, seig

    nvec = int(v.shape[0])
    nbasis = nvec - 1
    v0_all = np.asarray(v[0], dtype=np.float64).ravel()
    cand = np.where(np.abs(v0_all) >= float(root_v0_min))[0]
    if cand.size == 0:
        cand = np.asarray([int(np.argmax(np.abs(v0_all)))], dtype=np.int64)

    # Candidate filtering by step caps is optional (requires ngorb + xs).
    trust_enabled = (
        (trust_maxabs_orb is not None or trust_maxabs_ci is not None)
        and (ngorb is not None)
        and (int(ngorb) >= 0)
        and ax is not None
    )

    # Root homing overlap in the generalized eigenvector space (S-metric).
    v_prev_use: np.ndarray | None = None
    if v_prev is not None and root_homing:
        vp = np.asarray(v_prev, dtype=np.float64).ravel()
        if vp.size < nvec:
            vp = np.pad(vp, (0, nvec - vp.size))
        elif vp.size > nvec:
            vp = vp[:nvec]
        v_prev_use = vp

    # Predicted quadratic decrease in the AH model: g·x + 0.5 x·H x
    # computed in the small subspace using heff/ovlp blocks.
    dE: np.ndarray | None = None
    if root_pred_decrease and nbasis > 0:
        heff_b = np.asarray(heff[1:nvec, 1:nvec], dtype=np.float64)
        b = np.asarray(heff[1:nvec, 0], dtype=np.float64).ravel()
        s_full = np.asarray(ovlp[1:nvec, 1:nvec], dtype=np.float64)
        dE = np.full((int(v.shape[1]),), np.inf, dtype=np.float64)

        use_block = abs(float(mu_orb) - float(mu_ci)) > 0.0
        s_orb = s_ci = None
        if use_block:
            if ovlp_orb is not None and ovlp_ci is not None:
                s_orb = np.asarray(ovlp_orb[1:nvec, 1:nvec], dtype=np.float64)
                s_ci = np.asarray(ovlp_ci[1:nvec, 1:nvec], dtype=np.float64)
            else:
                use_block = False  # fallback to scalar-style correction

        for k in cand.tolist():
            v0k = float(v0_all[k])
            if abs(v0k) < 1e-14:
                continue
            ck = np.asarray(v[1:nvec, k], dtype=np.float64).ravel() * (1.0 / v0k)
            g_dot_x = float(np.dot(ck, b))
            xHx_tilde = float(ck @ (heff_b @ ck))

            # Convert from shifted operator (H+mu) back to the true AH quadratic model.
            if use_block and s_orb is not None and s_ci is not None:
                x2_orb = float(ck @ (s_orb @ ck))
                x2_ci = float(ck @ (s_ci @ ck))
                xHx = xHx_tilde - float(mu_orb) * x2_orb - float(mu_ci) * x2_ci
            else:
                x2 = float(ck @ (s_full @ ck))
                # If mu_orb != mu_ci but block overlaps are unavailable, apply the smaller
                # shift to avoid over-correcting (still monotone in mu).
                mu_eff = float(mu_orb) if abs(float(mu_orb) - float(mu_ci)) < 1e-15 else float(min(mu_orb, mu_ci))
                xHx = xHx_tilde - mu_eff * x2

            dE[k] = g_dot_x + 0.5 * xHx

    # Choose the root.
    sel = int(cand[0])
    if dE is not None:
        # Prefer candidates that satisfy the trust caps (check only a few best).
        order = cand[np.argsort(dE[cand], kind="stable")]
        cand_ok: list[int] = []
        if trust_enabled and ngorb is not None:
            ncheck = int(min(int(order.size), 5))
            for k in order[:ncheck].tolist():
                v0k = float(v0_all[k])
                if abs(v0k) < 1e-14:
                    continue
                xk = _dgemv(np.asarray(v[1:, k], dtype=np.float64).ravel() * (1.0 / v0k), xs)
                max_orb = float(np.max(np.abs(xk[: int(ngorb)]))) if int(ngorb) > 0 else 0.0
                max_ci = float(np.max(np.abs(xk[int(ngorb) :]))) if int(ngorb) < int(xk.size) else 0.0
                ok_orb = True if trust_maxabs_orb is None else (max_orb <= float(trust_maxabs_orb))
                ok_ci = True if trust_maxabs_ci is None else (max_ci <= float(trust_maxabs_ci))
                if ok_orb and ok_ci:
                    cand_ok.append(int(k))
            if cand_ok:
                order = np.asarray(cand_ok, dtype=np.int64)

        # Root homing as a tie-breaker among near-equivalent predicted decreases.
        if v_prev_use is not None and root_homing and order.size > 1:
            best = float(np.min(dE[order]))
            tol = max(1e-12, abs(best) * float(root_pred_decrease_tol_rel))
            near = [int(k) for k in order.tolist() if float(dE[k]) <= best + tol]
            if len(near) > 1:
                overlaps = []
                for k in near:
                    vk = np.asarray(v[:, k], dtype=np.float64).ravel()
                    overlaps.append(abs(float(v_prev_use.conj() @ (ovlp @ vk))))
                sel = int(near[int(np.argmax(overlaps))])
            else:
                sel = int(order[0])
        else:
            sel = int(order[0])
    elif v_prev_use is not None and root_homing:
        # Homing-only selection among candidates.
        overlaps = []
        for k in cand.tolist():
            vk = np.asarray(v[:, k], dtype=np.float64).ravel()
            overlaps.append(abs(float(v_prev_use.conj() @ (ovlp @ vk))))
        sel = int(cand[int(np.argmax(overlaps))])
    else:
        # Legacy: pick the first eigenvector with sufficient v0 component.
        sel = int(cand[0])

    log.debug1("CIAH eigen-sel %d", sel)
    w_t = float(w[sel])

    v0 = float(v[0, sel])
    if abs(v0) < 1e-14:
        return np.zeros_like(xs[0]), w_t, v[:, sel], sel, seig
    xtrial = _dgemv(np.asarray(v[1:, sel], dtype=np.float64).ravel() * (1.0 / v0), xs)
    return np.asarray(xtrial, dtype=np.float64), w_t, np.asarray(v[:, sel], dtype=np.float64), sel, seig


def davidson_cc(
    h_op: Callable[[np.ndarray], np.ndarray],
    g_op: Callable[[], np.ndarray],
    precond: Callable[[np.ndarray, float], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-10,
    xs: Sequence[np.ndarray] = (),
    ax: Sequence[np.ndarray] = (),
    max_cycle: int = 30,
    lindep: float = 1e-14,
    dot: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.dot,
    verbose: Any | None = None,
    *,
    root_v0_min: float = 0.1,
    root_homing: bool = False,
    root_pred_decrease: bool = False,
    root_pred_decrease_tol_rel: float = 1e-3,
    trust_maxabs_orb: float | None = None,
    trust_maxabs_ci: float | None = None,
    ngorb: int | None = None,
    mu_orb: float = 0.0,
    mu_ci: float = 0.0,
    mgs: bool = False,
    mgs_eps: float = 1e-12,
    restart: bool = False,
    restart_stagnant: int = 3,
) -> Iterator[tuple[bool, int, float, np.ndarray, np.ndarray, np.ndarray, float]]:
    """Internal AH-Davidson iterator (PySCF `ciah.davidson_cc` equivalent).

    Parameters
    ----------
    h_op : Callable[[np.ndarray], np.ndarray]
        Hamiltonian operator over the parameter vector.
    g_op : Callable[[], np.ndarray]
        Gradient operator (returns the gradient vector).
    precond : Callable[[np.ndarray, float], np.ndarray]
        Preconditioner function `P(x, e)`.
    x0 : np.ndarray
        Initial guess vector.
    tol : float, optional
        Convergence tolerance.
    xs : Sequence[np.ndarray], optional
        Initial subspace vectors.
    ax : Sequence[np.ndarray], optional
        Action of H on initial subspace vectors.
    max_cycle : int, optional
        Maximum number of cycles.
    lindep : float, optional
        Linear dependency threshold.
    dot : Callable[[np.ndarray, np.ndarray], np.ndarray], optional
        Dot product function.
    verbose : Any | None, optional
        Logger or verbosity level.

    Yields
    ------
    tuple
        (converged, cycle, energy, x_trial, hx, dx, min_eig)
    """

    log = _new_logger(verbose=verbose)
    toloose = float(np.sqrt(float(tol)))
    xs_l = [np.asarray(v, dtype=np.float64).ravel() for v in xs]
    ax_l = [np.asarray(v, dtype=np.float64).ravel() for v in ax]
    nx = int(len(xs_l))

    x0 = np.asarray(x0, dtype=np.float64).ravel()
    problem_size = int(x0.size)
    max_cycle = min(int(max_cycle), problem_size)

    heff = np.zeros((max_cycle + nx + 1, max_cycle + nx + 1), dtype=np.float64)
    ovlp = np.eye(max_cycle + nx + 1, dtype=np.float64)
    ovlp_orb = None
    ovlp_ci = None
    if root_pred_decrease and (abs(float(mu_orb) - float(mu_ci)) > 0.0) and ngorb is not None:
        # Block overlaps are only needed when mu differs across blocks.
        ovlp_orb = np.zeros_like(ovlp)
        ovlp_ci = np.zeros_like(ovlp)
    if nx == 0:
        xs_l.append(x0)
        ax_l.append(np.asarray(h_op(x0), dtype=np.float64).ravel())
    else:
        for i in range(1, nx + 1):
            for j in range(1, i + 1):
                heff[i, j] = float(dot(xs_l[i - 1].conj(), ax_l[j - 1]).real)
            ovlp[i, j] = float(dot(xs_l[i - 1].conj(), xs_l[j - 1]).real)
            heff[1:i, i] = heff[i, 1:i]
            ovlp[1:i, i] = ovlp[i, 1:i]
        if ovlp_orb is not None and ovlp_ci is not None and ngorb is not None:
            ng = int(ngorb)
            for i in range(1, nx + 1):
                for j in range(1, i + 1):
                    ovlp_orb[i, j] = float(dot(xs_l[i - 1][:ng].conj(), xs_l[j - 1][:ng]).real) if ng > 0 else 0.0
                    ovlp_ci[i, j] = float(dot(xs_l[i - 1][ng:].conj(), xs_l[j - 1][ng:]).real)
                ovlp_orb[1:i, i] = ovlp_orb[i, 1:i]
                ovlp_ci[1:i, i] = ovlp_ci[i, 1:i]

    w_t = 0.0
    v_prev: np.ndarray | None = None
    best_dx = float("inf")
    n_stagnant = 0
    for istep in range(max_cycle):
        g = np.asarray(g_op(), dtype=np.float64).ravel()
        nx = len(xs_l)
        for i in range(nx):
            heff[i + 1, 0] = float(dot(xs_l[i].conj(), g).real)
            heff[nx, i + 1] = float(dot(xs_l[nx - 1].conj(), ax_l[i]).real)
            ovlp[nx, i + 1] = float(dot(xs_l[nx - 1].conj(), xs_l[i]).real)
            if ovlp_orb is not None and ovlp_ci is not None and ngorb is not None:
                ng = int(ngorb)
                ovlp_orb[nx, i + 1] = float(dot(xs_l[nx - 1][:ng].conj(), xs_l[i][:ng]).real) if ng > 0 else 0.0
                ovlp_ci[nx, i + 1] = float(dot(xs_l[nx - 1][ng:].conj(), xs_l[i][ng:]).real)
        heff[0, : nx + 1] = heff[: nx + 1, 0].conj()
        heff[1:nx, nx] = heff[nx, 1:nx].conj()
        ovlp[1:nx, nx] = ovlp[nx, 1:nx].conj()
        if ovlp_orb is not None and ovlp_ci is not None:
            ovlp_orb[1:nx, nx] = ovlp_orb[nx, 1:nx].conj()
            ovlp_ci[1:nx, nx] = ovlp_ci[nx, 1:nx].conj()

        nvec = nx + 1
        wlast = w_t
        xtrial, w_t, v_t, index, seig = _regular_step(
            heff[:nvec, :nvec],
            ovlp[:nvec, :nvec],
            xs_l,
            ax_l if ax_l else None,
            lindep,
            log,
            v_prev=v_prev,
            root_v0_min=root_v0_min,
            root_homing=bool(root_homing),
            root_pred_decrease=bool(root_pred_decrease),
            root_pred_decrease_tol_rel=float(root_pred_decrease_tol_rel),
            trust_maxabs_orb=trust_maxabs_orb,
            trust_maxabs_ci=trust_maxabs_ci,
            ngorb=ngorb,
            mu_orb=float(mu_orb),
            mu_ci=float(mu_ci),
            ovlp_orb=ovlp_orb[:nvec, :nvec] if ovlp_orb is not None else None,
            ovlp_ci=ovlp_ci[:nvec, :nvec] if ovlp_ci is not None else None,
        )
        s0 = float(seig[0]) if seig.size else 0.0
        if v_t.size == 0:
            z = np.zeros_like(x0)
            yield True, istep + 1, w_t, z, z, z, s0
            break
        v_prev = np.asarray(v_t, dtype=np.float64).ravel()

        hx = _dgemv(v_t[1:], ax_l)
        dx = hx + g * float(v_t[0]) - w_t * float(v_t[0]) * xtrial
        norm_dx = float(np.linalg.norm(dx))
        log.debug1(
            "... AH step %d  index=%d  |dx|=%.5g  eig=%.5g  v[0]=%.5g  lindep=%.5g",
            istep + 1,
            index,
            norm_dx,
            w_t,
            float(v_t[0]) if v_t.size else 0.0,
            s0,
        )

        if abs(float(v_t[0])) > 1e-14:
            hx = hx * (1.0 / float(v_t[0]))
        else:
            hx = np.zeros_like(hx)

        if restart and norm_dx < best_dx * 0.999:
            best_dx = norm_dx
            n_stagnant = 0
        elif restart:
            n_stagnant += 1
            if n_stagnant >= int(restart_stagnant) and istep + 1 < max_cycle:
                # Restart the subspace with the current best direction (xtrial).
                log.debug1("AH Davidson restart at step %d (stagnation)", istep + 1)
                n_stagnant = 0
                best_dx = norm_dx
                v_prev = None
                x_reset = np.asarray(xtrial, dtype=np.float64).ravel()
                nrm = float(np.linalg.norm(x_reset))
                if nrm > 0.0:
                    x_reset *= 1.0 / nrm
                xs_l = [x_reset]
                ax_l = [np.asarray(h_op(x_reset), dtype=np.float64).ravel()]
                heff.fill(0.0)
                ovlp.fill(0.0)
                np.fill_diagonal(ovlp, 1.0)
                if ovlp_orb is not None:
                    ovlp_orb.fill(0.0)
                if ovlp_ci is not None:
                    ovlp_ci.fill(0.0)
                continue

        converged = (
            (abs(w_t - wlast) < float(tol) and norm_dx < toloose)
            or s0 < float(lindep)
            or (istep + 1 == problem_size)
        )
        if converged:
            yield True, istep + 1, w_t, xtrial, hx, dx, s0
            if s0 < float(lindep) or norm_dx < float(lindep):
                break
        else:
            yield False, istep + 1, w_t, xtrial, hx, dx, s0
            x1 = np.asarray(precond(dx, w_t), dtype=np.float64).ravel()
            if mgs and xs_l:
                x1_orig = x1.copy()
                for vj in xs_l:
                    x1 = x1 - float(dot(vj.conj(), x1).real) * vj
                nrm = float(np.linalg.norm(x1))
                if nrm < float(mgs_eps):
                    x1 = x1_orig
                    nrm = float(np.linalg.norm(x1))
                if nrm > 0.0:
                    x1 *= 1.0 / nrm
            xs_l.append(x1)
            ax_l.append(np.asarray(h_op(x1), dtype=np.float64).ravel())


@dataclass(frozen=True)
class WeightsInfo:
    """Resolved weights information for state-averaged calculations.

    Attributes
    ----------
    weights : np.ndarray
        Normalized weights, shape (nroots,).
    source : str
        Source of weights ("arg", "mc.weights", or "equal").
    mismatch : bool
        Whether there was a mismatch between mc.weights and fcisolver.weights.
    """
    weights: np.ndarray
    source: str
    mismatch: bool


@dataclass(frozen=True)
class CIActiveHamiltonian:
    """Active-space Hamiltonian pieces used by CI-side blocks.

    Attributes
    ----------
    h1cas : np.ndarray
        Core Hamiltonian in the active space (h1 + V_core), shape (ncas, ncas).
    eri_cas : np.ndarray
        Active-space 2-electron integrals, shape (ncas, ncas, ncas, ncas) in chemist ordering.
    """
    h1cas: np.ndarray
    eri_cas: np.ndarray


@dataclass(frozen=True)
class _PackedCI:
    ci0_list: list[np.ndarray]  # per-root flattened CI vectors
    pack: Callable[[Sequence[np.ndarray]], np.ndarray]
    unpack: Callable[[np.ndarray], list[np.ndarray]]


def _pack_ci_getters(ci0: Any) -> _PackedCI:
    ci0_list = _as_ci_list(ci0)
    if len(ci0_list) == 1:
        def _pack(x: Sequence[np.ndarray]) -> np.ndarray:
            return np.asarray(x[0], dtype=np.float64).ravel()

        def _unpack(x: np.ndarray) -> list[np.ndarray]:
            return [np.asarray(x, dtype=np.float64).ravel()]

        return _PackedCI(ci0_list=ci0_list, pack=_pack, unpack=_unpack)

    sizes = [int(np.asarray(c).size) for c in ci0_list]
    offs: list[int] = [0]
    for s in sizes[:-1]:
        offs.append(offs[-1] + int(s))
    total = int(sum(sizes))

    def _pack(x: Sequence[np.ndarray]) -> np.ndarray:
        parts = [np.asarray(v, dtype=np.float64).ravel() for v in x]
        return np.concatenate(parts, axis=0)

    def _unpack(x: np.ndarray) -> list[np.ndarray]:
        x = np.asarray(x, dtype=np.float64).ravel()
        if int(x.size) != total:
            raise ValueError("packed CI length mismatch")
        out: list[np.ndarray] = []
        off = 0
        for s in sizes:
            out.append(np.asarray(x[off: off + s], dtype=np.float64).copy())
            off += int(s)
        return out

    return _PackedCI(ci0_list=ci0_list, pack=_pack, unpack=_unpack)


@contextmanager
def _maybe_set_attr(obj: Any, name: str, value: Any) -> Iterator[bool]:
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
        if changed:
            try:
                if old is missing:
                    delattr(obj, name)
                else:
                    setattr(obj, name, old)
            except Exception:
                pass


@contextmanager
def _ah_mixed_precision_ctx(fcisolver: Any, enable: bool) -> Iterator[None]:
    """Temporarily set fcisolver.matvec_cuda_dtype='mixed' for AH contract_2e calls."""
    if not enable or fcisolver is None:
        yield
        return
    old = getattr(fcisolver, "matvec_cuda_dtype", "float64")
    try:
        fcisolver.matvec_cuda_dtype = "mixed"
        yield
    finally:
        fcisolver.matvec_cuda_dtype = old


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).ravel()
    if w.size == 0:
        raise ValueError("weights must be non-empty")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    s = float(np.sum(w))
    if s <= 0.0:
        raise ValueError("weights must sum to a positive number")
    return np.asarray(w / s, dtype=np.float64)


def _resolve_weights(
    mc: Any,
    *,
    nroots: int,
    weights: Sequence[float] | None,
    strict: bool,
) -> WeightsInfo:
    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be >= 1")

    mismatch = False
    if weights is not None:
        w = _normalize_weights(np.asarray(weights, dtype=np.float64))
        source = "arg"
    else:
        w_mc = getattr(mc, "weights", None)
        if w_mc is None:
            w = np.ones(nroots, dtype=np.float64) / float(nroots)
            source = "equal"
        else:
            w = _normalize_weights(np.asarray(w_mc, dtype=np.float64))
            source = "mc.weights"

    if int(w.size) != nroots:
        raise ValueError(f"weights must have length nroots={nroots}, got {int(w.size)}")

    # Cross-check against fcisolver.weights if available (PySCF SA wrappers store weights there).
    fs = getattr(mc, "fcisolver", None)
    w_fs = getattr(fs, "weights", None)
    if w_fs is not None:
        w_fs = _normalize_weights(np.asarray(w_fs, dtype=np.float64))
        if int(w_fs.size) == nroots and not np.allclose(w_fs, w, rtol=0.0, atol=1e-12):
            mismatch = True
            if strict:
                raise ValueError("mc.weights and mc.fcisolver.weights differ (strict mode)")

    return WeightsInfo(weights=w, source=source, mismatch=bool(mismatch))


def _as_ci_list(ci: Any) -> list[np.ndarray]:
    if isinstance(ci, np.ndarray):
        c = np.asarray(ci, dtype=np.float64).ravel()
        if c.size == 0:
            raise ValueError("empty CI vector")
        return [c]
    if isinstance(ci, (list, tuple)):
        out: list[np.ndarray] = []
        for c in ci:
            arr = np.asarray(c, dtype=np.float64).ravel()
            if arr.size == 0:
                raise ValueError("empty CI vector in list")
            out.append(arr)
        if len(out) == 0:
            raise ValueError("empty CI list")
        return out
    raise TypeError(f"unsupported CI type: {type(ci)!r}")


def pack_ci_list(ci_list: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate per-root CI vectors into a single packed 1D vector.

    Parameters
    ----------
    ci_list : Sequence[np.ndarray]
        List of CI vectors.

    Returns
    -------
    np.ndarray
        The packed CI vector.
    """

    parts = [np.asarray(c, dtype=np.float64).ravel() for c in ci_list]
    return np.concatenate(parts, axis=0)


def unpack_ci_list(x: np.ndarray, template_ci_list: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Unpack a packed CI vector to match the per-root shapes of `template_ci_list`.

    Parameters
    ----------
    x : np.ndarray
        Packed CI vector.
    template_ci_list : Sequence[np.ndarray]
        List of template CI vectors (defining shapes).

    Returns
    -------
    list[np.ndarray]
        List of unpacked CI vectors.
    """

    x = np.asarray(x, dtype=np.float64).ravel()
    sizes = [int(np.asarray(c).size) for c in template_ci_list]
    total = int(sum(sizes))
    if int(x.size) != total:
        raise ValueError(f"packed CI length mismatch: expected {total}, got {int(x.size)}")
    out: list[np.ndarray] = []
    off = 0
    for c, sz in zip(template_ci_list, sizes):
        arr = x[off : off + sz].copy()
        out.append(arr.reshape(np.asarray(c).shape))
        off += sz
    return out


def compute_ci_gram_inv(ci_list: Sequence[np.ndarray]) -> np.ndarray:
    """Return (C^T C)^(-1) for CI root columns C=[c1,...,cR].

    Parameters
    ----------
    ci_list : Sequence[np.ndarray]
        List of CI vectors.

    Returns
    -------
    np.ndarray
        Inverse Gram matrix, shape (nroots, nroots).
    """

    c_list = [np.asarray(c, dtype=np.float64).ravel() for c in ci_list]
    nroots = int(len(c_list))
    if nroots == 0:
        raise ValueError("empty CI list")
    nci = int(c_list[0].size)
    if any(int(c.size) != nci for c in c_list):
        raise ValueError("inconsistent CI sizes across roots")

    cmat = np.stack(c_list, axis=1)  # (nci, nroots)
    gram = cmat.T @ cmat
    try:
        return np.linalg.inv(gram)
    except np.linalg.LinAlgError:  # pragma: no cover
        return np.linalg.pinv(gram)


def project_ci_root_span(
    ci_ref_list: Sequence[np.ndarray],
    vec_list: Sequence[np.ndarray],
    *,
    gram_inv: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Project vectors to the orthogonal complement of the CI root span.

    Implements `v <- v - C (C^T C)^(-1) C^T v` for `C=[c1,...,cR]`.

    Parameters
    ----------
    ci_ref_list : Sequence[np.ndarray]
        Reference CI vectors defining the span.
    vec_list : Sequence[np.ndarray]
        Vectors to project.
    gram_inv : np.ndarray | None, optional
        Precomputed inverse Gram matrix.

    Returns
    -------
    list[np.ndarray]
        Projected vectors.
    """

    ci_ref = [np.asarray(c, dtype=np.float64).ravel() for c in ci_ref_list]
    vecs = [np.asarray(v, dtype=np.float64) for v in vec_list]
    if len(vecs) != len(ci_ref):
        raise ValueError("vec_list must have the same length as ci_ref_list")
    nroots = int(len(ci_ref))
    nci = int(ci_ref[0].size)
    if any(int(c.size) != nci for c in ci_ref):
        raise ValueError("inconsistent CI sizes across roots")

    if gram_inv is None:
        gram_inv_use = compute_ci_gram_inv(ci_ref)
    else:
        gram_inv_use = np.asarray(gram_inv, dtype=np.float64)
        if gram_inv_use.shape != (nroots, nroots):
            raise ValueError("gram_inv has wrong shape")

    cmat = np.stack(ci_ref, axis=1)  # (nci, nroots)
    out: list[np.ndarray] = []
    for v in vecs:
        shape = v.shape
        vflat = v.ravel()
        if int(vflat.size) != nci:
            raise ValueError("CI vector size mismatch in projection")
        coeff = cmat.T @ vflat  # (nroots,)
        vproj = vflat - cmat @ (gram_inv_use @ coeff)
        out.append(np.ascontiguousarray(vproj.reshape(shape)))
    return out


def _build_ci_active_hamiltonian(casscf: Any, mo: np.ndarray, eris: Any) -> CIActiveHamiltonian:
    """Build (h1cas, eri_cas) for the Newton-CASSCF operator.

    Parameters
    ----------
    casscf : Any
        CASSCF object.
    mo : np.ndarray
        Molecular orbitals.
    eris : Any
        Integral object.

    Returns
    -------
    CIActiveHamiltonian
        The active space Hamiltonian components.
    """

    ncas = int(getattr(casscf, "ncas"))
    ncore = int(getattr(casscf, "ncore"))
    nocc = ncore + ncas
    if ncas <= 0:
        raise ValueError("ncas must be positive")

    hcore = casscf.get_hcore()
    h1e_mo = _to_np_f64((mo.T @ hcore) @ mo)

    vhf_c = getattr(eris, "vhf_c", None)
    if vhf_c is None:
        raise ValueError("eris must provide attribute 'vhf_c'")
    vhf_c = _to_np_f64(vhf_c)

    h1cas = np.asarray(h1e_mo[ncore:nocc, ncore:nocc], dtype=np.float64) + np.asarray(
        vhf_c[ncore:nocc, ncore:nocc], dtype=np.float64
    )

    ppaa = getattr(eris, "ppaa", None)
    if ppaa is None:
        raise ValueError("eris must provide attribute 'ppaa' (needed for eri_cas)")
    # PySCF builds eri_cas[a] = ppaa[p=a+ncore][q in active]
    try:
        eri_cas = _to_np_f64(ppaa)[ncore:nocc, ncore:nocc]
    except Exception:
        eri_cas = np.empty((ncas, ncas, ncas, ncas), dtype=np.float64)
        for p in range(ncore, nocc):
            eri_cas[p - ncore] = _to_np_f64(ppaa)[p][ncore:nocc]

    return CIActiveHamiltonian(h1cas=h1cas, eri_cas=eri_cas)


def _maybe_gen_linkstr(fcisolver: Any, ncas: int, nelecas: Any, tril: bool) -> Any | None:
    """Best-effort determinant link table for contract_2e speedups (optional)."""

    gen_linkstr = getattr(fcisolver, "gen_linkstr", None)
    if gen_linkstr is None:
        return None
    try:
        return gen_linkstr(int(ncas), nelecas, bool(tril))
    except Exception:
        return None


def _ci_h_op(
    fcisolver: Any,
    *,
    h1cas: np.ndarray,
    eri_cas: np.ndarray,
    ncas: int,
    nelecas: Any,
    ci_list: Sequence[np.ndarray],
    link_index: Any | None,
) -> list[np.ndarray]:
    """Return [H_act @ ci_i] for each root, flattened.

    Parameters
    ----------
    fcisolver : Any
        FCI solver object.
    h1cas : np.ndarray
        Core Hamiltonian in active space.
    eri_cas : np.ndarray
        Active space ERIs.
    ncas : int
        Number of active orbitals.
    nelecas : Any
        Number of electrons in active space.
    ci_list : Sequence[np.ndarray]
        List of CI vectors.
    link_index : Any | None
        Determinant link table.

    Returns
    -------
    list[np.ndarray]
        List of H @ ci vectors (flattened).
    """

    op = fcisolver.absorb_h1e(h1cas, eri_cas, int(ncas), nelecas, 0.5)
    out: list[np.ndarray] = []
    for c in ci_list:
        hc = fcisolver.contract_2e(op, c, int(ncas), nelecas, link_index=link_index)
        out.append(np.asarray(hc, dtype=np.float64).ravel())
    return out


def _ci_h_diag(
    fcisolver: Any,
    *,
    h1cas: np.ndarray,
    eri_cas: np.ndarray,
    ncas: int,
    nelecas: Any,
) -> np.ndarray:
    """Return diag(H_act) as a flat 1D vector.

    Parameters
    ----------
    fcisolver : Any
        FCI solver object.
    h1cas : np.ndarray
        Core Hamiltonian in active space.
    eri_cas : np.ndarray
        Active space ERIs.
    ncas : int
        Number of active orbitals.
    nelecas : Any
        Number of electrons in active space.

    Returns
    -------
    np.ndarray
        Diagonal of the active space Hamiltonian.
    """

    hd = fcisolver.make_hdiag(h1cas, eri_cas, int(ncas), nelecas)
    return np.asarray(hd, dtype=np.float64).ravel()


def _compute_ci_grad_and_diag_blocks(
    casscf: Any,
    mo: np.ndarray,
    ci0: Any,
    eris: Any,
    *,
    weights: Sequence[float] | None,
    gauge: str,
    strict_weights: bool,
    enforce_absorb_h1e_direct: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (g_ci, hdiag_ci) blocks in standard packing and scaling.

    Parameters
    ----------
    casscf : Any
        CASSCF object.
    mo : np.ndarray
        Molecular orbitals.
    ci0 : Any
        Reference CI vector(s).
    eris : Any
        Integral object.
    weights : Sequence[float] | None
        State-average weights.
    gauge : str
        Gauge for CI projection ("none", "project", "project_out").
    strict_weights : bool
        Whether to enforce weight consistency.
    enforce_absorb_h1e_direct : bool
        Whether to force direct H1e absorption.

    Returns
    -------
    g_ci : np.ndarray
        Packed CI gradient block, including global factor-2 scaling.
    hdiag_ci : np.ndarray
        Packed CI diagonal block (preconditioner), including global factor-2 scaling.

    Notes
    -----
    - This routine is intended for incremental parity development. It does not
      attempt to reproduce the full orbital blocks in `newton_casscf.gen_g_hop`.
    - For SA (multi-root) inputs and `gauge != "none"`, this function applies
      root-span projection to the returned CI gradient block.
    """

    ci_list = _as_ci_list(ci0)
    nroots = int(len(ci_list))
    ncas = int(getattr(casscf, "ncas"))
    ncore = int(getattr(casscf, "ncore"))
    nocc = ncore + ncas
    nelecas = getattr(casscf, "nelecas")

    w_info = _resolve_weights(casscf, nroots=nroots, weights=weights, strict=bool(strict_weights))

    fcisolver = getattr(casscf, "fcisolver", None)
    if fcisolver is None:
        raise ValueError("casscf must provide fcisolver to compute CI blocks")

    if enforce_absorb_h1e_direct:
        ctx_absorb = _maybe_set_attr(fcisolver, "absorb_h1e_mode", "direct")
    else:
        ctx_absorb = nullcontext(False)

    gauge_l = str(gauge).strip().lower()
    if gauge_l not in ("none", "project", "project_out"):
        raise ValueError("gauge must be one of: none, project, project_out")

    # Build active-space Hamiltonian pieces at the reference.
    ham = _build_ci_active_hamiltonian(casscf, mo, eris)

    with ctx_absorb:
        linkstrl = _maybe_gen_linkstr(fcisolver, ncas, nelecas, True)
        hci0 = _ci_h_op(
            fcisolver,
            h1cas=ham.h1cas,
            eri_cas=ham.eri_cas,
            ncas=ncas,
            nelecas=nelecas,
            ci_list=ci_list,
            link_index=linkstrl,
        )
        eci0 = np.asarray([float(np.dot(c, hc)) for c, hc in zip(ci_list, hci0)], dtype=np.float64)
        gci_resid = [hc - c * float(e) for hc, c, e in zip(hci0, ci_list, eci0)]

        # PySCF scales CI gradients by weights and then applies a global factor 2.
        gci_w = [g * float(w) for g, w in zip(gci_resid, w_info.weights)]

        if nroots > 1 and gauge_l in ("project", "project_out"):
            gci_w = project_ci_root_span(ci_list, gci_w)

        g_ci = pack_ci_list(gci_w) * 2.0

        # PySCF's CI diagonal includes an intermediate-normalization correction.
        hd0 = _ci_h_diag(fcisolver, h1cas=ham.h1cas, eri_cas=ham.eri_cas, ncas=ncas, nelecas=nelecas)
        hci_diag = [hd0 - float(e) - g * c * 2.0 for hd0, e, g, c in zip([hd0] * nroots, eci0, gci_resid, ci_list)]
        hci_diag = [h * float(w) for h, w in zip(hci_diag, w_info.weights)]
        hdiag_ci = pack_ci_list(hci_diag) * 2.0

    # Sanity on packed sizes
    n_ci_total = int(sum(int(np.asarray(c).size) for c in ci_list))
    if int(g_ci.size) != n_ci_total or int(hdiag_ci.size) != n_ci_total:
        raise RuntimeError("internal error: packed CI block sizes mismatch")

    return np.asarray(g_ci, dtype=np.float64).ravel(), np.asarray(hdiag_ci, dtype=np.float64).ravel()


def _compute_ci_cc_matvec_block(
    casscf: Any,
    mo: np.ndarray,
    ci0: Any,
    eris: Any,
    ci1: Any,
    *,
    weights: Sequence[float] | None,
    gauge: str,
    strict_weights: bool,
    enforce_absorb_h1e_direct: bool,
) -> np.ndarray:
    """Compute the CI output block of PySCF's `h_op` for CI-only directions.

    This implements the `H_cc` portion of PySCF `newton_casscf.h_op` (including the
    intermediate-normalization correction terms) and applies SA weights and the
    global factor-2 scaling.

    It is intended as an incremental building block: for an input direction
    `x = [0; x_ci]`, PySCF's `h_op(x)` has CI output equal to this function's
    return value (up to optional gauge projection).

    Parameters
    ----------
    casscf : Any
        CASSCF object.
    mo : np.ndarray
        Molecular orbitals.
    ci0 : Any
        Reference CI vector(s).
    eris : Any
        Integral object.
    ci1 : Any
        CI direction vector(s).
    weights : Sequence[float] | None
        State-average weights.
    gauge : str
        Gauge for CI projection.
    strict_weights : bool
        Whether to check weights consistency.
    enforce_absorb_h1e_direct : bool
        Whether to force absorb_h1e=direct.

    Returns
    -------
    np.ndarray
        The computed matrix-vector product block.
    """

    ci0_list = _as_ci_list(ci0)
    nroots = int(len(ci0_list))

    # Accept ci1 as list/tuple (per-root) or as a packed vector.
    if isinstance(ci1, (list, tuple)):
        ci1_list = [np.asarray(c, dtype=np.float64).ravel() for c in ci1]
    else:
        ci1_arr = np.asarray(ci1, dtype=np.float64)
        if nroots == 1:
            ci1_list = [ci1_arr.ravel()]
        else:
            ci1_list = unpack_ci_list(ci1_arr.ravel(), ci0_list)

    if len(ci1_list) != nroots:
        raise ValueError("ci1 must have the same number of roots as ci0")

    w_info = _resolve_weights(casscf, nroots=nroots, weights=weights, strict=bool(strict_weights))

    fcisolver = getattr(casscf, "fcisolver", None)
    if fcisolver is None:
        raise ValueError("casscf must provide fcisolver to compute CI blocks")

    if enforce_absorb_h1e_direct:
        ctx_absorb = _maybe_set_attr(fcisolver, "absorb_h1e_mode", "direct")
    else:
        ctx_absorb = nullcontext(False)

    gauge_l = str(gauge).strip().lower()
    if gauge_l not in ("none", "project", "project_out"):
        raise ValueError("gauge must be one of: none, project, project_out")

    if nroots > 1 and gauge_l == "project":
        ci1_list = project_ci_root_span(ci0_list, ci1_list)

    ncas = int(getattr(casscf, "ncas"))
    nelecas = getattr(casscf, "nelecas")
    ham = _build_ci_active_hamiltonian(casscf, mo, eris)

    with ctx_absorb:
        linkstrl = _maybe_gen_linkstr(fcisolver, ncas, nelecas, True)
        hci0 = _ci_h_op(
            fcisolver,
            h1cas=ham.h1cas,
            eri_cas=ham.eri_cas,
            ncas=ncas,
            nelecas=nelecas,
            ci_list=ci0_list,
            link_index=linkstrl,
        )
        eci0 = np.asarray([float(np.dot(c, hc)) for c, hc in zip(ci0_list, hci0)], dtype=np.float64)
        gci0 = [hc - c * float(e) for hc, c, e in zip(hci0, ci0_list, eci0)]

        hci1 = _ci_h_op(
            fcisolver,
            h1cas=ham.h1cas,
            eri_cas=ham.eri_cas,
            ncas=ncas,
            nelecas=nelecas,
            ci_list=ci1_list,
            link_index=linkstrl,
        )

    out_list: list[np.ndarray] = []
    for c0, c1, hc1, ec0, gc0, w in zip(ci0_list, ci1_list, hci1, eci0, gci0, w_info.weights):
        # Fix intermediate normalization? Mirrors PySCF 2023/09/15 update in `newton_casscf.h_op`.
        v = hc1 - c1 * float(ec0)
        v = v - gc0 * float(np.dot(c0, c1))
        v = v - c0 * float(np.dot(gc0, c1))
        out_list.append(np.asarray(v, dtype=np.float64) * float(w))

    if nroots > 1 and gauge_l in ("project", "project_out"):
        out_list = project_ci_root_span(ci0_list, out_list)

    return np.asarray(pack_ci_list(out_list) * 2.0, dtype=np.float64).ravel()


def _compute_ci_co_matvec_block(
    casscf: Any,
    mo: np.ndarray,
    ci0: Any,
    eris: Any,
    x_orb: np.ndarray,
    *,
    weights: Sequence[float] | None,
    gauge: str,
    strict_weights: bool,
    enforce_absorb_h1e_direct: bool,
) -> np.ndarray:
    """Compute the CI output block due to an orbital-direction input (H_co, CI part).

    This matches the `kci0` term in PySCF's `newton_casscf.h_op`:

      kci0 = H_act'[x_orb] @ ci0  (projected to remove per-root norm direction)

    and then applies SA weights and the global factor-2 scaling.

    Parameters
    ----------
    casscf : Any
        CASSCF object.
    mo : np.ndarray
        Molecular orbitals.
    ci0 : Any
        Reference CI vector(s).
    eris : Any
        Integral object.
    x_orb : np.ndarray
        Orbital rotation parameters (packed).
    weights : Sequence[float] | None
        State-average weights.
    gauge : str
        Gauge for CI projection.
    strict_weights : bool
        Whether to check weights consistency.
    enforce_absorb_h1e_direct : bool
        Whether to force absorb_h1e=direct.

    Returns
    -------
    np.ndarray
        The computed H_co block.

    Notes
    -----
    This helper assumes the CI direction part of the input vector is zero; it only
    computes the CI output induced by the orbital-rotation direction.
    """

    ci0_list = _as_ci_list(ci0)
    nroots = int(len(ci0_list))

    w_info = _resolve_weights(casscf, nroots=nroots, weights=weights, strict=bool(strict_weights))

    fcisolver = getattr(casscf, "fcisolver", None)
    if fcisolver is None:
        raise ValueError("casscf must provide fcisolver to compute CI blocks")

    if enforce_absorb_h1e_direct:
        ctx_absorb = _maybe_set_attr(fcisolver, "absorb_h1e_mode", "direct")
    else:
        ctx_absorb = nullcontext(False)

    gauge_l = str(gauge).strip().lower()
    if gauge_l not in ("none", "project", "project_out"):
        raise ValueError("gauge must be one of: none, project, project_out")

    ncas = int(getattr(casscf, "ncas"))
    ncore = int(getattr(casscf, "ncore"))
    nocc = ncore + ncas
    nelecas = getattr(casscf, "nelecas")
    nmo = int(mo.shape[1])

    x_orb = np.asarray(x_orb, dtype=np.float64).ravel()
    x1 = casscf.unpack_uniq_var(x_orb) if int(x_orb.size) else np.zeros((nmo, nmo), dtype=np.float64)

    rc = np.asarray(x1[:, :ncore], dtype=np.float64, order="C") if ncore else np.zeros((nmo, 0), dtype=np.float64)
    ra = np.asarray(x1[:, ncore:nocc], dtype=np.float64, order="C")

    ddm_c = np.zeros((nmo, nmo), dtype=np.float64)
    if ncore:
        ddm_c[:, :ncore] = rc[:, :ncore] * 2.0
        ddm_c[:ncore, :] += rc[:, :ncore].T * 2.0

    hcore = casscf.get_hcore()
    h1e_mo = _to_np_f64((mo.T @ hcore) @ mo)
    vhf_c = getattr(eris, "vhf_c", None)
    if vhf_c is None:
        raise ValueError("eris must provide attribute 'vhf_c'")
    vhf_c = _to_np_f64(vhf_c)

    # jk response from core density variation (PySCF's in-loop accumulation).
    jk = np.zeros((ncas, ncas), dtype=np.float64)
    if ncore:
        ppaa = getattr(eris, "ppaa", None)
        papa = getattr(eris, "papa", None)
        if ppaa is None or papa is None:
            raise ValueError("eris must provide 'ppaa' and 'papa' for H_co")
        xp, _on_gpu = _get_xp(ppaa, papa)
        ppaa_g = _to_xp_f64(ppaa, xp)
        papa_g = _to_xp_f64(papa, xp)
        ddm_c_g = xp.asarray(ddm_c, dtype=xp.float64)
        jk = _to_np_f64(
            xp.einsum("iq,iquv->uv", ddm_c_g, ppaa_g, optimize=True)
            - 0.5 * xp.einsum("iq,iuqv->uv", ddm_c_g, papa_g, optimize=True)
        )

    # First-order active-space Hamiltonian pieces induced by orbital rotation x1.
    ppaa = getattr(eris, "ppaa", None)
    if ppaa is None:
        raise ValueError("eris must provide attribute 'ppaa' for H_co")
    # paaa contraction on GPU
    xp_ci, _ = _get_xp(ppaa)
    paaa_g = _to_xp_f64(ppaa, xp_ci)[:, ncore:nocc]
    ra_g = xp_ci.asarray(ra, dtype=xp_ci.float64)

    aaaa = _to_np_f64(
        (ra_g.T @ paaa_g.reshape(nmo, -1)).reshape((ncas, ncas, ncas, ncas))
    )
    aaaa = aaaa + aaaa.transpose(1, 0, 2, 3)
    aaaa = aaaa + aaaa.transpose(2, 3, 0, 1)

    h1row = np.asarray(h1e_mo[ncore:nocc] + np.asarray(vhf_c[ncore:nocc], dtype=np.float64), dtype=np.float64)
    h1aa = h1row @ ra
    h1aa = h1aa + h1aa.T + jk

    with ctx_absorb:
        linkstrl = _maybe_gen_linkstr(fcisolver, ncas, nelecas, True)
        op = fcisolver.absorb_h1e(h1aa, aaaa, int(ncas), nelecas, 0.5)
        out_list: list[np.ndarray] = []
        for c0, w in zip(ci0_list, w_info.weights):
            kc0 = fcisolver.contract_2e(op, c0, int(ncas), nelecas, link_index=linkstrl)
            kc0 = np.asarray(kc0, dtype=np.float64).ravel()
            kc0 = kc0 - float(np.dot(kc0, c0)) * c0
            out_list.append(kc0 * float(w))

    if nroots > 1 and gauge_l in ("project", "project_out"):
        out_list = project_ci_root_span(ci0_list, out_list)

    return np.asarray(pack_ci_list(out_list) * 2.0, dtype=np.float64).ravel()


def _build_gpq_per_root(
    casscf: Any,
    mo: np.ndarray,
    ci0_list: Sequence[np.ndarray],
    eris: Any,
    *,
    strict_weights: bool = False,
) -> np.ndarray:
    """Build per-root gpq matrices (as in PySCF `newton_casscf.gen_g_hop`).

    Notes
    -----
    - This follows standard Newton-CASSCF conventions and is intended for parity testing and for
      implementing overlap-correction terms (e.g. in H_oc).
    - The result is *per-root* (unweighted). SA weights are applied by callers.
    """

    ncas = int(getattr(casscf, "ncas"))
    ncore = int(getattr(casscf, "ncore"))
    nocc = ncore + ncas
    nelecas = getattr(casscf, "nelecas")
    nmo = int(mo.shape[1])
    nroots = int(len(ci0_list))
    if nroots <= 0:
        raise ValueError("ci0_list must be non-empty")

    # Trigger the weights consistency check early (even though gpq itself is per-root).
    _resolve_weights(casscf, nroots=nroots, weights=None, strict=bool(strict_weights))

    fcisolver = getattr(casscf, "fcisolver", None)
    if fcisolver is None:
        raise ValueError("casscf must provide fcisolver to build gpq")

    link_index = _maybe_gen_linkstr(fcisolver, ncas, nelecas, False)
    try:
        casdm1, casdm2 = fcisolver.states_make_rdm12(ci0_list, ncas, nelecas, link_index=link_index)
        casdm1 = np.asarray(casdm1, dtype=np.float64)
        casdm2 = np.asarray(casdm2, dtype=np.float64)
    except AttributeError:
        casdm1_list: list[np.ndarray] = []
        casdm2_list: list[np.ndarray] = []
        for c in ci0_list:
            dm1, dm2 = fcisolver.make_rdm12(c, ncas, nelecas, link_index=link_index)
            casdm1_list.append(np.asarray(dm1, dtype=np.float64))
            casdm2_list.append(np.asarray(dm2, dtype=np.float64))
        casdm1 = np.asarray(casdm1_list, dtype=np.float64)
        casdm2 = np.asarray(casdm2_list, dtype=np.float64)

    if casdm1.shape != (nroots, ncas, ncas):
        raise RuntimeError("unexpected casdm1 shape in gpq builder")
    if casdm2.shape != (nroots, ncas, ncas, ncas, ncas):
        raise RuntimeError("unexpected casdm2 shape in gpq builder")

    ppaa = getattr(eris, "ppaa", None)
    papa = getattr(eris, "papa", None)
    if ppaa is None or papa is None:
        raise ValueError("eris must provide 'ppaa' and 'papa' for gpq builder")
    vhf_c = getattr(eris, "vhf_c", None)
    if vhf_c is None:
        raise ValueError("eris must provide 'vhf_c' for gpq builder")

    xp, _on_gpu = _get_xp(ppaa, papa)
    vhf_c_np = _to_np_f64(vhf_c)

    ppaa_g = _to_xp_f64(ppaa, xp)
    papa_g = _to_xp_f64(papa, xp)
    casdm1_g = xp.asarray(casdm1, dtype=xp.float64)
    casdm2_g = xp.asarray(casdm2, dtype=xp.float64)

    vhf_a = xp.einsum("pquv,ruv->rpq", ppaa_g, casdm1_g, optimize=True)
    vhf_a -= 0.5 * xp.einsum("puqv,ruv->rpq", papa_g, casdm1_g, optimize=True)

    jtmp_full = xp.einsum("pquv,ruvwx->rpqwx", ppaa_g, casdm2_g, optimize=True)
    g_dm2 = _to_np_f64(xp.einsum("rpuuv->rpv", jtmp_full[:, :, ncore:nocc, :, :], optimize=True))

    vhf_ca = _to_np_f64(vhf_a) + vhf_c_np[None, :, :]

    hcore = casscf.get_hcore()
    # Ensure mo and hcore are on the same device before matmul.
    h1e_mo = _to_np_f64((_to_xp_f64(mo, xp).T @ _to_xp_f64(hcore, xp)) @ _to_xp_f64(mo, xp))

    gpq = np.zeros((nroots, nmo, nmo), dtype=np.float64)
    if ncore:
        gpq[:, :, :ncore] = (h1e_mo[None, :, :ncore] + vhf_ca[:, :, :ncore]) * 2.0

    tmp = h1e_mo[:, ncore:nocc] + vhf_c_np[:, ncore:nocc]
    gpq[:, :, ncore:nocc] = np.dot(tmp, casdm1).transpose(1, 0, 2)
    gpq[:, :, ncore:nocc] += g_dm2

    return gpq


def _compute_orb_grad_block_from_gpq(
    casscf: Any,
    mo: np.ndarray,
    ci0: Any,
    eris: Any,
    *,
    weights: Sequence[float] | None,
    strict_weights: bool,
) -> np.ndarray:
    """Compute the packed orbital-gradient block (standard scaling) via per-root gpq.

    Returns
    -------
    g_orb_vec
        Packed unique orbital gradient vector, including the global factor-2 scaling
        used by PySCF `newton_casscf.gen_g_hop`.
    """

    ci0_list = _as_ci_list(ci0)
    nroots = int(len(ci0_list))
    w_info = _resolve_weights(casscf, nroots=nroots, weights=weights, strict=bool(strict_weights))

    gpq = _build_gpq_per_root(casscf, mo, ci0_list, eris, strict_weights=bool(strict_weights))
    g_orb_mat = np.einsum("r,rpq->pq", w_info.weights, gpq, optimize=True)
    g_orb_vec = casscf.pack_uniq_var(g_orb_mat - g_orb_mat.T)
    return np.asarray(g_orb_vec, dtype=np.float64).ravel() * 2.0


def _weighted_trans_rdm12(
    fcisolver: Any,
    *,
    ci1_list: Sequence[np.ndarray],
    ci0_list: Sequence[np.ndarray],
    weights: np.ndarray,
    ncas: int,
    nelecas: Any,
    link_index: Any | None,
    return_cupy: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return SA-weighted (tdm1,tdm2) built from per-root transitions."""

    if return_cupy:
        try:
            import cupy as _cp  # type: ignore[import-not-found]
        except Exception:
            _cp = None
    else:
        _cp = None
    xp = _cp if _cp is not None and return_cupy else np

    nroots = int(len(ci0_list))
    if nroots != int(len(ci1_list)):
        raise ValueError("ci1_list length mismatch")
    _rdm_kw: dict = dict(link_index=link_index, return_cupy=return_cupy)
    if return_cupy:
        _rdm_kw["rdm_backend"] = "cuda"
    if nroots == 1:
        dm1, dm2 = fcisolver.trans_rdm12(ci1_list[0], ci0_list[0], int(ncas), nelecas, **_rdm_kw)
        return xp.asarray(dm1, dtype=xp.float64), xp.asarray(dm2, dtype=xp.float64)

    states_trans = getattr(fcisolver, "states_trans_rdm12", None)
    tdm1 = xp.zeros((ncas, ncas), dtype=xp.float64)
    tdm2 = xp.zeros((ncas, ncas, ncas, ncas), dtype=xp.float64)
    if callable(states_trans) and not return_cupy:
        dm1s, dm2s = states_trans(ci1_list, ci0_list, int(ncas), nelecas, link_index=link_index)
        for w, dm1, dm2 in zip(weights, dm1s, dm2s):
            tdm1 += float(w) * xp.asarray(dm1, dtype=xp.float64)
            tdm2 += float(w) * xp.asarray(dm2, dtype=xp.float64)
    else:
        for w, c1, c0 in zip(weights, ci1_list, ci0_list):
            dm1, dm2 = fcisolver.trans_rdm12(c1, c0, int(ncas), nelecas, **_rdm_kw)
            tdm1 += float(w) * xp.asarray(dm1, dtype=xp.float64)
            tdm2 += float(w) * xp.asarray(dm2, dtype=xp.float64)
    return tdm1, tdm2


@dataclass(frozen=True)
class _NewtonInternalCache:
    ncas: int
    ncore: int
    nocc: int
    nmo: int
    nroots: int
    nelecas: Any
    weights: np.ndarray

    ci: _PackedCI

    h1e_mo: np.ndarray
    gpq: np.ndarray  # (nroots,nmo,nmo) per-root
    vhf_c: np.ndarray  # (nmo,nmo) core Coulomb/exchange (eris.vhf_c)
    vhf_ca: np.ndarray  # (nmo,nmo) SA-averaged
    casdm1: np.ndarray  # (ncas,ncas) SA-averaged
    jkcaa: np.ndarray  # (nocc,ncas) SA-averaged
    hdm2: np.ndarray  # (nmo,ncas,nmo,ncas) SA-averaged
    paaa: np.ndarray  # (nmo,ncas,ncas,ncas) from eris.ppaa[:,act]
    paaa_gpu: Any  # same as paaa but on GPU (CuPy) when available, else numpy
    dm1_full: np.ndarray  # (nmo,nmo) full (core+active) density used in H_oo

    h1cas_0: np.ndarray  # (ncas,ncas)
    eri_cas: np.ndarray  # (ncas,ncas,ncas,ncas)
    hci0: list[np.ndarray]  # per-root flattened
    eci0: np.ndarray  # (nroots,)
    gci0: list[np.ndarray]  # per-root residual vectors (unweighted)

    hdiag_all: np.ndarray  # packed diag (orb+ci), scaled by 2
    g_all: np.ndarray  # packed gradient, scaled by 2
    ngorb: int


def _build_internal_cache(
    casscf: Any,
    mo: np.ndarray,
    ci0: Any,
    eris: Any,
    *,
    weights: Sequence[float] | None,
    strict_weights: bool,
    enforce_absorb_h1e_direct: bool,
    ah_mixed_precision: bool = False,
) -> _NewtonInternalCache:
    ci = _pack_ci_getters(ci0)
    ci0_list = ci.ci0_list
    nroots = int(len(ci0_list))

    ncas = int(getattr(casscf, "ncas"))
    ncore = int(getattr(casscf, "ncore"))
    nocc = ncore + ncas
    nelecas = getattr(casscf, "nelecas")
    nmo = int(mo.shape[1])

    w_info = _resolve_weights(casscf, nroots=nroots, weights=weights, strict=bool(strict_weights))
    w = np.asarray(w_info.weights, dtype=np.float64)

    fcisolver = getattr(casscf, "fcisolver", None)
    if fcisolver is None:
        raise ValueError("casscf must provide fcisolver")

    if enforce_absorb_h1e_direct:
        ctx_absorb = _maybe_set_attr(fcisolver, "absorb_h1e_mode", "direct")
    else:
        ctx_absorb = nullcontext(False)

    with ctx_absorb:
        linkstr = _maybe_gen_linkstr(fcisolver, ncas, nelecas, False)

        try:
            casdm1_r, casdm2_r = fcisolver.states_make_rdm12(ci0_list, ncas, nelecas, link_index=linkstr)
            casdm1_r = np.asarray(casdm1_r, dtype=np.float64)
            casdm2_r = np.asarray(casdm2_r, dtype=np.float64)
        except AttributeError:
            dm1s: list[np.ndarray] = []
            dm2s: list[np.ndarray] = []
            for c in ci0_list:
                dm1, dm2 = fcisolver.make_rdm12(c, ncas, nelecas, link_index=linkstr)
                dm1s.append(np.asarray(dm1, dtype=np.float64))
                dm2s.append(np.asarray(dm2, dtype=np.float64))
            casdm1_r = np.asarray(dm1s, dtype=np.float64)
            casdm2_r = np.asarray(dm2s, dtype=np.float64)

    ppaa = getattr(eris, "ppaa", None)
    papa = getattr(eris, "papa", None)
    if ppaa is None or papa is None:
        raise ValueError("eris must provide 'ppaa' and 'papa'")

    # Detect array backend from eris — keep contractions on GPU if available.
    xp, _on_gpu = _get_xp(ppaa, papa)

    ppaa_arr = _to_xp_f64(ppaa, xp)  # (nmo,nmo,ncas,ncas)
    papa_arr = _to_xp_f64(papa, xp)  # (nmo,ncas,nmo,ncas)

    # Upload RDMs to same device
    casdm1_g = xp.asarray(casdm1_r, dtype=xp.float64)
    casdm2_g = xp.asarray(casdm2_r, dtype=xp.float64)
    dm2tmp_g = casdm2_g.transpose(0, 2, 3, 1, 4) + casdm2_g.transpose(0, 1, 3, 2, 4)

    # paaa[p,u,v,w] = ppaa[p, ncore:nocc, u, v]
    paaa = ppaa_arr[:, ncore:nocc, :, :]  # (nmo,ncas,ncas,ncas)

    # eri_cas = ppaa[ncore:nocc, ncore:nocc]
    eri_cas = xp.asarray(ppaa_arr[ncore:nocc, ncore:nocc], dtype=xp.float64)

    # jkcaa_r
    arange_nocc = xp.arange(nocc)
    ppaa_diag = ppaa_arr[arange_nocc, arange_nocc]  # (nocc,ncas,ncas)
    papa_diag = papa_arr[arange_nocc, :, arange_nocc]  # (nocc,ncas,ncas)
    jkcaa_kernel = 6.0 * papa_diag - 2.0 * ppaa_diag  # (nocc,ncas,ncas)
    jkcaa_r = xp.einsum("pik,rik->rpi", jkcaa_kernel, casdm1_g, optimize=True)

    # vhf_a_r
    vhf_a_r = xp.einsum("pquv,ruv->rpq", ppaa_arr, casdm1_g, optimize=True)
    vhf_a_r -= 0.5 * xp.einsum("puqv,ruv->rpq", papa_arr, casdm1_g, optimize=True)

    # hdm2_r and g_dm2_r — vectorized over p
    # jtmp[r,p,q,u,v] = ppaa[p,q,u',v'] * dm2[r,u'v',uv]
    # = einsum('pquv,ruvwx->rpqwx', ppaa, casdm2_r)
    # hdm2_r[r,p,u,q,v] = jtmp[r,p,q,u,v] + ktmp[r,p,q,u,v]
    # where ktmp uses papa transposed
    # g_dm2_r[r,p,v] = sum_u jtmp[r,p,u_act,u_act,v]
    #
    # jtmp_full[r,p,q,w,x] = sum_{u,v} ppaa[p,q,u,v] * dm2[r,u,v,w,x]
    # Use GEMM: reshape to (nmo*nmo, ncas*ncas) @ (ncas*ncas, ncas*ncas)
    _ppaa_2d = ppaa_arr.reshape(nmo * nmo, ncas * ncas)
    _dm2_2d = casdm2_g.reshape(nroots, ncas * ncas, ncas * ncas)
    jtmp_full = xp.stack([_ppaa_2d @ _dm2_2d[r] for r in range(nroots)]).reshape(
        nroots, nmo, nmo, ncas, ncas
    )
    papa_t = papa_arr.transpose(0, 2, 1, 3)  # (nmo,nmo,ncas,ncas)
    _papa_t_2d = papa_t.reshape(nmo * nmo, ncas * ncas)
    _dm2tmp_2d = dm2tmp_g.reshape(nroots, ncas * ncas, ncas * ncas)
    ktmp_full = xp.stack([_papa_t_2d @ _dm2tmp_2d[r] for r in range(nroots)]).reshape(
        nroots, nmo, nmo, ncas, ncas
    )

    # hdm2_r[r,p,u,q,v] = (jtmp_full + ktmp_full)[r,p,q,u,v].transpose(0,1,3,2,4)
    hdm2_r = (jtmp_full + ktmp_full).transpose(0, 1, 3, 2, 4)

    # g_dm2_r[r,p,v] = sum_u jtmp_full[r,p,u_act,u_act,v]
    g_dm2_r = xp.einsum("rpuuv->rpv", jtmp_full[:, :, ncore:nocc, :, :], optimize=True)

    vhf_c = getattr(eris, "vhf_c", None)
    if vhf_c is None:
        raise ValueError("eris must provide 'vhf_c'")
    vhf_c = _to_xp_f64(vhf_c, xp)
    vhf_ca_r = vhf_c[None, :, :] + vhf_a_r

    hcore = casscf.get_hcore()
    h1e_mo = _to_np_f64((_to_xp_f64(mo, xp).T @ _to_xp_f64(hcore, xp)) @ _to_xp_f64(mo, xp))

    # Download to CPU for gpq assembly (mixed with CI quantities)
    vhf_ca_r_np = _to_np_f64(vhf_ca_r)
    vhf_c_np = _to_np_f64(vhf_c)
    g_dm2_r_np = _to_np_f64(g_dm2_r)

    gpq = np.zeros((nroots, nmo, nmo), dtype=np.float64)
    if ncore:
        gpq[:, :, :ncore] = (h1e_mo[None, :, :ncore] + vhf_ca_r_np[:, :, :ncore]) * 2.0
    gpq[:, :, ncore:nocc] = np.dot(h1e_mo[:, ncore:nocc] + vhf_c_np[:, ncore:nocc], casdm1_r).transpose(1, 0, 2)
    gpq[:, :, ncore:nocc] += g_dm2_r_np

    w_xp = xp.asarray(w, dtype=xp.float64)
    vhf_ca = _to_np_f64(xp.einsum("r,rpq->pq", w_xp, vhf_ca_r, optimize=True))
    casdm1 = np.einsum("r,rpq->pq", w, casdm1_r, optimize=True)
    jkcaa = _to_np_f64(xp.einsum("r,rpq->pq", w_xp, jkcaa_r, optimize=True))
    hdm2 = _to_np_f64(xp.einsum("r,rpqst->pqst", w_xp, hdm2_r, optimize=True))

    # Active-space Hamiltonian at reference (CPU for CI solver).
    h1cas_0 = h1e_mo[ncore:nocc, ncore:nocc] + vhf_c_np[ncore:nocc, ncore:nocc]
    eri_cas_np = _to_np_f64(eri_cas)
    paaa_np = _to_np_f64(paaa)
    # Keep GPU copy for einsum contractions in _h_op_raw
    paaa_gpu = _to_xp_f64(paaa, xp)

    if enforce_absorb_h1e_direct:
        ctx_absorb2 = _maybe_set_attr(fcisolver, "absorb_h1e_mode", "direct")
    else:
        ctx_absorb2 = nullcontext(False)

    with _ah_mixed_precision_ctx(fcisolver, ah_mixed_precision), ctx_absorb2:
        linkstrl = _maybe_gen_linkstr(fcisolver, ncas, nelecas, True)
        hci0 = _ci_h_op(
            fcisolver,
            h1cas=h1cas_0,
            eri_cas=eri_cas_np,
            ncas=ncas,
            nelecas=nelecas,
            ci_list=ci0_list,
            link_index=linkstrl,
        )
        eci0 = np.asarray([float(np.dot(c, hc)) for c, hc in zip(ci0_list, hci0)], dtype=np.float64)
        gci0 = [hc - c * float(e) for hc, c, e in zip(hci0, ci0_list, eci0)]

        # Orbital gradient block (via gpq) and CI gradient block.
        g_orb_mat = np.einsum("r,rpq->pq", w, gpq, optimize=True)
        g_orb = casscf.pack_uniq_var(g_orb_mat - g_orb_mat.T)
        ngorb = int(np.asarray(g_orb).size)
        g_ci = ci.pack([g * float(wi) for g, wi in zip(gci0, w)])
        g_all = np.hstack(
            (np.asarray(g_orb, dtype=np.float64).ravel() * 2.0, np.asarray(g_ci, dtype=np.float64).ravel() * 2.0)
        )

        # Orbital diagonal (PySCF parts 7-6).
        dm1_full = np.zeros((nmo, nmo), dtype=np.float64)
        if ncore:
            idx = np.arange(ncore)
            dm1_full[idx, idx] = 2.0
        dm1_full[ncore:nocc, ncore:nocc] = casdm1
        h_diag = np.einsum("ii,jj->ij", h1e_mo, dm1_full) - h1e_mo * dm1_full
        h_diag = h_diag + h_diag.T
        g_diag = np.einsum("r,rpp->p", w, gpq, optimize=True)
        h_diag -= g_diag + g_diag.reshape(-1, 1)
        idx = np.arange(nmo)
        h_diag[idx, idx] += g_diag * 2.0

        v_diag = np.diag(vhf_ca)
        h_diag[:, :ncore] += v_diag.reshape(-1, 1) * 2.0
        h_diag[:ncore] += v_diag * 2.0
        if ncore:
            idxc = np.arange(ncore)
            h_diag[idxc, idxc] -= v_diag[:ncore] * 4.0

        tmp = np.einsum("ii,jj->ij", vhf_c_np, casdm1, optimize=True)
        h_diag[:, ncore:nocc] += tmp
        h_diag[ncore:nocc, :] += tmp.T
        tmp2 = -vhf_c_np[ncore:nocc, ncore:nocc] * casdm1
        h_diag[ncore:nocc, ncore:nocc] += tmp2 + tmp2.T

        tmp3 = 6.0 * _to_np_f64(getattr(eris, "k_pc")) - 2.0 * _to_np_f64(getattr(eris, "j_pc"))
        h_diag[ncore:, :ncore] += tmp3[ncore:]
        h_diag[:ncore, ncore:] += tmp3[ncore:].T

        h_diag[:nocc, ncore:nocc] -= jkcaa
        h_diag[ncore:nocc, :nocc] -= jkcaa.T

        v_diag2 = np.einsum("ijij->ij", hdm2, optimize=True)
        h_diag[ncore:nocc, :] += v_diag2.T
        h_diag[:, ncore:nocc] += v_diag2

        h_diag = casscf.pack_uniq_var(h_diag)

        # CI diagonal (PySCF intermediate-normalization fix).
        hd0 = _ci_h_diag(fcisolver, h1cas=h1cas_0, eri_cas=eri_cas_np, ncas=ncas, nelecas=nelecas)
        hci_diag = [hd0 - float(ec) - gc * c * 2.0 for ec, gc, c in zip(eci0, gci0, ci0_list)]
        hci_diag = [h * float(wi) for h, wi in zip(hci_diag, w)]
        hdiag_all = np.hstack((np.asarray(h_diag, dtype=np.float64).ravel() * 2.0, ci.pack(hci_diag) * 2.0))

    return _NewtonInternalCache(
        ncas=ncas,
        ncore=ncore,
        nocc=nocc,
        nmo=nmo,
        nroots=nroots,
        nelecas=nelecas,
        weights=w,
        ci=ci,
        h1e_mo=h1e_mo,
        gpq=gpq,
        vhf_c=vhf_c_np,
        vhf_ca=vhf_ca,
        casdm1=casdm1,
        jkcaa=jkcaa,
        hdm2=hdm2,
        paaa=paaa_np,
        paaa_gpu=paaa_gpu,
        dm1_full=dm1_full,
        h1cas_0=h1cas_0,
        eri_cas=eri_cas_np,
        hci0=hci0,
        eci0=eci0,
        gci0=gci0,
        hdiag_all=np.asarray(hdiag_all, dtype=np.float64).ravel(),
        g_all=np.asarray(g_all, dtype=np.float64).ravel(),
        ngorb=int(ngorb),
    )


def _compute_orb_oc_matvec_block(
    casscf: Any,
    mo: np.ndarray,
    ci0: Any,
    eris: Any,
    ci1: Any,
    *,
    weights: Sequence[float] | None,
    gauge: str,
    strict_weights: bool,
    enforce_absorb_h1e_direct: bool,
    require_zero_s10: bool = True,
) -> np.ndarray:
    """Compute the orbital output block due to a CI-direction input (H_oc, orbital side).

    This helper targets the special case `x_orb = 0`. It mirrors the `H_oc` portion of
    PySCF `newton_casscf.h_op`:

      x2 += core-column contributions from vhf_a (built from transition 1-RDM)
      x2[:,act] += (h1e_mo + vhf_c) @ tdm1 + g_dm2 (from transition 2-RDM)
      x2 -= Σ_r s10_r * gpq_r     (overlap correction; optional)

    If `require_zero_s10=True`, this function raises unless the CI direction is in the
    gauge-fixed tangent space (`c_r^T δc_r = 0`), so the overlap correction term
    `x2 -= Σ_r s10_r * gpq_r` vanishes.  If `require_zero_s10=False`, this routine builds
    the per-root `gpq` matrices internally (PySCF parity path) and includes the overlap
    correction.
    """

    ci0_list = _as_ci_list(ci0)
    nroots = int(len(ci0_list))

    # Accept ci1 as list/tuple (per-root) or as packed vector.
    if isinstance(ci1, (list, tuple)):
        ci1_list = [np.asarray(c, dtype=np.float64).ravel() for c in ci1]
    else:
        ci1_arr = np.asarray(ci1, dtype=np.float64)
        if nroots == 1:
            ci1_list = [ci1_arr.ravel()]
        else:
            ci1_list = unpack_ci_list(ci1_arr.ravel(), ci0_list)

    if len(ci1_list) != nroots:
        raise ValueError("ci1 must have the same number of roots as ci0")

    w_info = _resolve_weights(casscf, nroots=nroots, weights=weights, strict=bool(strict_weights))

    fcisolver = getattr(casscf, "fcisolver", None)
    if fcisolver is None:
        raise ValueError("casscf must provide fcisolver to compute H_oc")

    if enforce_absorb_h1e_direct:
        ctx_absorb = _maybe_set_attr(fcisolver, "absorb_h1e_mode", "direct")
    else:
        ctx_absorb = nullcontext(False)

    gauge_l = str(gauge).strip().lower()
    if gauge_l not in ("none", "project", "project_out"):
        raise ValueError("gauge must be one of: none, project, project_out")

    if nroots > 1 and gauge_l == "project":
        ci1_list = project_ci_root_span(ci0_list, ci1_list)

    ncas = int(getattr(casscf, "ncas"))
    ncore = int(getattr(casscf, "ncore"))
    nocc = ncore + ncas
    nelecas = getattr(casscf, "nelecas")
    nmo = int(mo.shape[1])

    hcore = casscf.get_hcore()
    h1e_mo = _to_np_f64((mo.T @ hcore) @ mo)
    vhf_c = getattr(eris, "vhf_c", None)
    if vhf_c is None:
        raise ValueError("eris must provide attribute 'vhf_c'")
    vhf_c = _to_np_f64(vhf_c)

    linkstr = _maybe_gen_linkstr(fcisolver, ncas, nelecas, False)
    # For SA/multi-root, do not rely on wrapper-weighted `trans_rdm12`. Always build
    # per-root transition densities (if possible) and apply our resolved weights
    # explicitly so that weights have a single source of truth (supports GUGAFCISolver
    # without requiring PySCF's StateAverageFCISolver wrapper).
    with ctx_absorb:
        if nroots == 1:
            tdm1, tdm2 = fcisolver.trans_rdm12(
                ci1_list[0], ci0_list[0], int(ncas), nelecas, link_index=linkstr
            )
            tdm1 = np.asarray(tdm1, dtype=np.float64)
            tdm2 = np.asarray(tdm2, dtype=np.float64)
        else:
            tdm1_list: list[np.ndarray] = []
            tdm2_list: list[np.ndarray] = []
            states_trans = getattr(fcisolver, "states_trans_rdm12", None)
            if callable(states_trans):
                dm1s, dm2s = states_trans(ci1_list, ci0_list, int(ncas), nelecas, link_index=linkstr)
                for dm1, dm2 in zip(dm1s, dm2s):
                    tdm1_list.append(np.asarray(dm1, dtype=np.float64))
                    tdm2_list.append(np.asarray(dm2, dtype=np.float64))
            else:
                for c1, c0 in zip(ci1_list, ci0_list):
                    dm1, dm2 = fcisolver.trans_rdm12(c1, c0, int(ncas), nelecas, link_index=linkstr)
                    tdm1_list.append(np.asarray(dm1, dtype=np.float64))
                    tdm2_list.append(np.asarray(dm2, dtype=np.float64))

            tdm1 = np.zeros((ncas, ncas), dtype=np.float64)
            tdm2 = np.zeros((ncas, ncas, ncas, ncas), dtype=np.float64)
            for w, dm1, dm2 in zip(w_info.weights, tdm1_list, tdm2_list):
                tdm1 += float(w) * dm1
                tdm2 += float(w) * dm2
    tdm1 = tdm1 + tdm1.T
    tdm2 = tdm2 + tdm2.transpose(1, 0, 3, 2)
    tdm2 = (tdm2 + tdm2.transpose(2, 3, 0, 1)) * 0.5

    # Transition-driven core-column response (vhf_a) and active-column response (g_dm2).
    ppaa = getattr(eris, "ppaa", None)
    papa = getattr(eris, "papa", None)
    if ppaa is None or papa is None:
        raise ValueError("eris must provide 'ppaa' and 'papa' for H_oc")

    xp, _on_gpu = _get_xp(ppaa, papa)
    vhf_a = np.zeros((nmo, ncore), dtype=np.float64)
    if ncore:
        ppaa_g = _to_xp_f64(ppaa, xp)
        papa_g = _to_xp_f64(papa, xp)
        tdm1_g = xp.asarray(tdm1, dtype=xp.float64)
        vhf_a = _to_np_f64(
            xp.einsum("pquv,uv->pq", ppaa_g[:, :ncore], tdm1_g, optimize=True)
            - 0.5 * xp.einsum("puqv,uv->pq", papa_g[:, :, :ncore], tdm1_g, optimize=True)
        )

    paaa_g = _to_xp_f64(ppaa, xp)[:, ncore:nocc]  # (nmo,ncas,ncas,ncas)
    tdm2_g = xp.asarray(tdm2, dtype=xp.float64)
    g_dm2 = _to_np_f64(xp.einsum("puwx,wxuv->pv", paaa_g, tdm2_g, optimize=True))

    # Overlap correction term uses per-root gpq. For now, require s10 == 0.
    s10 = np.asarray([float(np.dot(c1, c0)) * 2.0 * float(w) for c1, c0, w in zip(ci1_list, ci0_list, w_info.weights)])
    if require_zero_s10 and float(np.max(np.abs(s10))) > 1e-10:
        raise NotImplementedError("H_oc overlap correction requires per-root gpq; pass a projected CI direction (s10≈0)")

    x2 = np.zeros((nmo, nmo), dtype=np.float64)
    if ncore:
        x2[:, :ncore] += (vhf_a) * 2.0
    x2[:, ncore:nocc] += (h1e_mo[:, ncore:nocc] + np.asarray(vhf_c[:, ncore:nocc], dtype=np.float64)) @ tdm1
    x2[:, ncore:nocc] += g_dm2

    if not require_zero_s10:
        sum_s10 = float(np.sum(s10))
        if ncore and abs(sum_s10) > 0.0:
            x2[:, :ncore] += (h1e_mo[:, :ncore] + np.asarray(vhf_c[:, :ncore], dtype=np.float64)) * (sum_s10 * 2.0)

        if float(np.max(np.abs(s10))) > 0.0:
            gpq = _build_gpq_per_root(
                casscf,
                mo,
                ci0_list,
                eris,
                strict_weights=bool(strict_weights),
            )
            x2 = x2 - np.einsum("r,rpq->pq", s10, gpq, optimize=True)

    x2 = x2 - x2.T
    return np.asarray(casscf.pack_uniq_var(x2) * 2.0, dtype=np.float64).ravel()


def _wrap_h_op_ci_projection(
    h_op: Callable[[np.ndarray], np.ndarray],
    *,
    n_orb: int,
    ci_ref_list: Sequence[np.ndarray],
    project_input: bool,
    project_output: bool,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a packed-vector h_op to apply CI root-span projection on input/output."""

    ci_ref = [np.asarray(c, dtype=np.float64) for c in ci_ref_list]
    gram_inv = compute_ci_gram_inv(ci_ref)
    n_ci_total = int(sum(int(np.asarray(c).size) for c in ci_ref))
    n_tot = int(n_orb + n_ci_total)

    def _call(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if int(x.size) != n_tot:
            raise ValueError("h_op input length mismatch")
        x_orb = x[:n_orb]
        x_ci = x[n_orb:]
        if project_input:
            x_ci_list = unpack_ci_list(x_ci, ci_ref)
            x_ci_list = project_ci_root_span(ci_ref, x_ci_list, gram_inv=gram_inv)
            x = np.concatenate([x_orb, pack_ci_list(x_ci_list)])

        y = np.asarray(h_op(x), dtype=np.float64).ravel()
        if int(y.size) != n_tot:
            raise ValueError("h_op output length mismatch")
        if project_output:
            y_orb = y[:n_orb]
            y_ci_list = unpack_ci_list(y[n_orb:], ci_ref)
            y_ci_list = project_ci_root_span(ci_ref, y_ci_list, gram_inv=gram_inv)
            y = np.concatenate([y_orb, pack_ci_list(y_ci_list)])
        return y

    return _call


def gen_g_hop_internal(
    casscf: Any,
    mo: np.ndarray,
    ci0: Any,
    eris: Any,
    verbose: int | None = None,
    *,
    weights: Sequence[float] | None = None,
    gauge: str = "none",
    convention: str = "pyscf2",
    strict_weights: bool = False,
    enforce_absorb_h1e_direct: bool = True,
    ah_mixed_precision: bool = False,
) -> tuple[np.ndarray, Callable[..., Any], Callable[[np.ndarray], np.ndarray], Any]:
    """cuGUGA-owned `gen_g_hop` implementation.

    Parameters
    ----------
    casscf : Any
        CASSCF object.
    mo : np.ndarray
        Molecular orbitals.
    ci0 : Any
        Reference CI vector(s).
    eris : Any
        Integral object.
    verbose : int | None, optional
        Verbosity level.
    weights : Sequence[float] | None, optional
        State-average weights.
    gauge : str, optional
        Gauge regarding CI projection. Defaults to "none".
    convention : str, optional
        Conventions version. Defaults to "pyscf2".
    strict_weights : bool, optional
        Whether to enforce weight consistency. Defaults to False.
    enforce_absorb_h1e_direct : bool, optional
        Whether to force direct H1e absorption. Defaults to True.

    Returns
    -------
    tuple
        (g, g_update, h_op, h_diag)
    """

    convention_l = str(convention).strip().lower()
    if convention_l not in ("pyscf2",):
        raise ValueError("unsupported convention (internal supports convention='pyscf2' only)")

    cache = _build_internal_cache(
        casscf,
        mo,
        ci0,
        eris,
        weights=weights,
        strict_weights=bool(strict_weights),
        enforce_absorb_h1e_direct=bool(enforce_absorb_h1e_direct),
        ah_mixed_precision=bool(ah_mixed_precision),
    )

    fcisolver = getattr(casscf, "fcisolver", None)
    if fcisolver is None:
        raise ValueError("casscf must provide fcisolver")

    def _absorb_ctx() -> Any:
        if enforce_absorb_h1e_direct:
            return _maybe_set_attr(fcisolver, "absorb_h1e_mode", "direct")
        return nullcontext(False)

    g_all = cache.g_all
    ngorb = int(cache.ngorb)
    ci0_list = cache.ci.ci0_list
    nroots = int(cache.nroots)

    def g_update(u: np.ndarray, fcivec: Any) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray], Any]:
        u = np.asarray(u, dtype=np.float64)
        if u.ndim != 2 or u.shape[0] != cache.nmo or u.shape[1] != cache.nmo:
            raise ValueError("u must be (nmo,nmo)")
        xp_mo, _ = _get_xp(mo)
        mo1 = _to_xp_f64(mo, xp_mo) @ xp_mo.asarray(u, dtype=xp_mo.float64)
        eris1 = casscf.ao2mo(mo1)
        g1, _gup, hop1, diag1 = gen_g_hop_internal(
            casscf,
            mo1,
            fcivec,
            eris1,
            verbose=verbose,
            weights=cache.weights,
            gauge="none",
            convention=convention,
            strict_weights=bool(strict_weights),
            enforce_absorb_h1e_direct=bool(enforce_absorb_h1e_direct),
            ah_mixed_precision=bool(ah_mixed_precision),
        )
        return np.asarray(g1, dtype=np.float64).ravel(), hop1, diag1

    with _absorb_ctx():
        linkstrl = _maybe_gen_linkstr(fcisolver, cache.ncas, cache.nelecas, True)
        with _ah_mixed_precision_ctx(fcisolver, ah_mixed_precision):
            op_h0 = fcisolver.absorb_h1e(cache.h1cas_0, cache.eri_cas, cache.ncas, cache.nelecas, 0.5)
    linkstr = _maybe_gen_linkstr(fcisolver, cache.ncas, cache.nelecas, False)

    # ── Closure-scope setup for _h_op_raw ──
    _ppaa_hop = getattr(eris, "ppaa", None)
    _papa_hop = getattr(eris, "papa", None)
    if _ppaa_hop is None or _papa_hop is None:
        raise ValueError("eris must provide 'ppaa' and 'papa'")

    # Detect GPU mode from eris integral storage.
    _hop_xp, _hop_on_gpu = _get_xp(_ppaa_hop, _papa_hop)
    _supports_return_gpu = hasattr(casscf, "df_B")

    # DF factors for memory-efficient per-iteration contractions.
    _L_pu_hop = getattr(eris, "L_pu", None)
    _L_pi_hop = getattr(eris, "L_pi", None)
    _use_df_factors = _L_pu_hop is not None

    if _use_df_factors:
        # Store smaller DF factors; skip ppaa/papa for per-iteration work.
        if _hop_on_gpu:
            L_pu_dev = _to_xp_f64(_L_pu_hop, _hop_xp)
            L_pi_dev = _to_xp_f64(_L_pi_hop, _hop_xp) if _L_pi_hop is not None else None
        L_pu_cpu = _to_np_f64(_L_pu_hop)
        L_pi_cpu = _to_np_f64(_L_pi_hop) if _L_pi_hop is not None else None
        # ppaa/papa are NOT captured for the closure.
        ppaa_cpu = None
        papa_cpu = None
    else:
        # CPU copies (always needed for fallback / CPU-only callers).
        ppaa_cpu = _to_np_f64(_ppaa_hop)
        papa_cpu = _to_np_f64(_papa_hop)

    # GPU copies of tensors used inside _h_op_raw (one-time upload).
    if _hop_on_gpu:
        if not _use_df_factors:
            ppaa_dev = _to_xp_f64(_ppaa_hop, _hop_xp)
            papa_dev = _to_xp_f64(_papa_hop, _hop_xp)
        else:
            ppaa_dev = None
            papa_dev = None
        ci0_list_dev = [_hop_xp.asarray(c, dtype=_hop_xp.float64).ravel() for c in ci0_list]
        hci0_dev = [_hop_xp.asarray(h, dtype=_hop_xp.float64).ravel() for h in cache.hci0]
        eci0_dev = _hop_xp.asarray(cache.eci0, dtype=_hop_xp.float64)
        h1e_mo_dev = _hop_xp.asarray(cache.h1e_mo, dtype=_hop_xp.float64)
        vhf_c_dev = _hop_xp.asarray(cache.vhf_c, dtype=_hop_xp.float64)
        vhf_ca_dev = _hop_xp.asarray(cache.vhf_ca, dtype=_hop_xp.float64)
        casdm1_dev = _hop_xp.asarray(cache.casdm1, dtype=_hop_xp.float64)
        hdm2_dev = _hop_xp.asarray(cache.hdm2, dtype=_hop_xp.float64)
        dm1_full_dev = _hop_xp.asarray(cache.dm1_full, dtype=_hop_xp.float64)
        gpq_dev = _hop_xp.asarray(cache.gpq, dtype=_hop_xp.float64)
        weights_dev = _hop_xp.asarray(cache.weights, dtype=_hop_xp.float64)
        paaa_dev = cache.paaa_gpu if cache.paaa_gpu is not None else _hop_xp.asarray(cache.paaa, dtype=_hop_xp.float64)
        # CI unpack offsets
        _ci_sizes = [int(c.size) for c in ci0_list_dev]
        _ci_offs: list[int] = [0]
        for _s in _ci_sizes[:-1]:
            _ci_offs.append(_ci_offs[-1] + _s)
        # Rebuild op_h0 with GPU h1e *and* GPU eri_cas for zero-sync contract_2e.
        h1cas_0_dev = _hop_xp.asarray(cache.h1cas_0, dtype=_hop_xp.float64)
        eri_cas_dev = _hop_xp.asarray(cache.eri_cas, dtype=_hop_xp.float64)
        with _absorb_ctx():
            with _ah_mixed_precision_ctx(fcisolver, ah_mixed_precision):
                op_h0_dev = fcisolver.absorb_h1e(h1cas_0_dev, eri_cas_dev, cache.ncas, cache.nelecas, 0.5)

    import os as _os_hop
    _HOP_PROFILE = _os_hop.environ.get("ASUKA_HOP_PROFILE", "0") == "1"

    # ── Precompute MO-basis 3-index DF integrals L_pq^Q for fast JK in h_op ──
    # L[p,q,Q] = sum_mn C[m,p] * B[m,n,Q] * C[n,q]
    # This replaces AO-basis _df_JK calls (nao^3 * naux) with MO-basis (nmo^3 * naux).
    _L_pq_dev = None
    _L_t_dev = None  # (nmo, naux, nmo) contiguous — avoids transpose+copy in h_op
    _use_mo_jk = False
    _disable_mo_jk = _os_hop.environ.get("ASUKA_DISABLE_MO_JK", "0") == "1"
    if not _disable_mo_jk and _hop_on_gpu and _supports_return_gpu and cache.ncore > 0:
        _df_B_raw = getattr(casscf, "df_B", None)
        if _df_B_raw is not None:
            _xp = _hop_xp
            _B = _xp.asarray(_df_B_raw, dtype=_xp.float64)
            _C = _xp.asarray(mo, dtype=_xp.float64)
            _nao_B = int(_B.shape[0])
            _nmo_C = int(_C.shape[1])
            _naux_B = int(_B.shape[2])
            # Half-transform: H[p,n,Q] = C.T @ B[:,:,Q] for all Q
            _H = (_C.T @ _B.reshape(_nao_B, _nao_B * _naux_B)).reshape(_nmo_C, _nao_B, _naux_B)
            # Full transform: L_t[p,Q,q] = sum_n H[p,n,Q] * C[n,q]
            # Transpose H to (nmo, naux, nao) so reshape to (nmo*naux, nao) preserves
            # the grouping: row p*naux+Q contains H[p, :, Q].
            _H_t = _xp.ascontiguousarray(_H.transpose(0, 2, 1))  # (nmo, naux, nao)
            _L_t_dev = (_H_t.reshape(_nmo_C * _naux_B, _nao_B) @ _C).reshape(_nmo_C, _naux_B, _nmo_C)
            _L_t_dev = _xp.ascontiguousarray(_L_t_dev)  # ensure contiguous
            # Also store (nmo, nmo, naux) layout for J einsum
            _L_pq_dev = _xp.ascontiguousarray(_L_t_dev.transpose(0, 2, 1))
            del _H, _B
            _use_mo_jk = True

    def _h_op_raw(x):
        # Use GPU path when GPU tensors exist in closure, regardless of
        # input vector type.  davidson_cc passes NumPy vectors, but the
        # expensive operations (sigma vectors, trans-RDMs, einsums over
        # ppaa/papa) benefit hugely from running on GPU.  The parameter
        # vector upload (~25K doubles) is negligible.
        if _HOP_PROFILE:
            _t0_hop = time.perf_counter()
        if _hop_on_gpu:
            xp = _hop_xp
            on_gpu = True
            x = xp.asarray(x, dtype=xp.float64).ravel()
        else:
            xp, on_gpu = _get_xp(x)
            x = xp.asarray(x, dtype=xp.float64).ravel()

        # Select device-appropriate tensors.
        ppaa = papa = None  # may stay None when using DF factors
        if on_gpu:
            paaa = paaa_dev
            if _use_df_factors:
                _L_pu, _L_pi = L_pu_dev, L_pi_dev
            else:
                ppaa, papa = ppaa_dev, papa_dev
            _ci0 = ci0_list_dev
            _hci0 = hci0_dev
            _eci0 = eci0_dev
            _h1e_mo = h1e_mo_dev
            _vhf_c = vhf_c_dev
            _vhf_ca = vhf_ca_dev
            _casdm1 = casdm1_dev
            _hdm2 = hdm2_dev
            _dm1_full = dm1_full_dev
            _gpq = gpq_dev
            _weights = weights_dev
            _op_h0 = op_h0_dev
        else:
            paaa = cache.paaa
            if _use_df_factors:
                _L_pu, _L_pi = L_pu_cpu, L_pi_cpu
            else:
                ppaa, papa = ppaa_cpu, papa_cpu
            _ci0 = ci0_list
            _hci0 = cache.hci0
            _eci0 = cache.eci0
            _h1e_mo = cache.h1e_mo
            _vhf_c = cache.vhf_c
            _vhf_ca = cache.vhf_ca
            _casdm1 = cache.casdm1
            _hdm2 = cache.hdm2
            _dm1_full = cache.dm1_full
            _gpq = cache.gpq
            _weights = cache.weights
            _op_h0 = op_h0

        x1 = casscf.unpack_uniq_var(x[:ngorb])  # xp-aware (Step 5)

        # CI unpack (xp-aware).
        ci_flat = x[ngorb:]
        if nroots == 1:
            ci1_list = [ci_flat.copy()]
        else:
            ci1_list = [ci_flat[off:off + sz].copy() for off, sz in zip(_ci_offs, _ci_sizes)] if on_gpu else cache.ci.unpack(ci_flat)

        # ── CI Hessian: H0|c1> ──
        if _HOP_PROFILE:
            if on_gpu: xp.cuda.Stream.null.synchronize()
            _t1_hop = time.perf_counter()
        _c2e_kw: dict = dict(link_index=linkstrl, return_cupy=on_gpu)
        if on_gpu:
            _c2e_kw["contract_2e_backend"] = "cuda"
        with _ah_mixed_precision_ctx(fcisolver, ah_mixed_precision), _absorb_ctx():
            hci1 = [
                fcisolver.contract_2e(_op_h0, c1, cache.ncas, cache.nelecas, **_c2e_kw).ravel()
                for c1 in ci1_list
            ]
        # Intermediate-normalisation correction (zero float() calls on GPU).
        hci1 = [hc1 - c1 * ec0 for hc1, c1, ec0 in zip(hci1, ci1_list, _eci0)]
        hci1 = [
            hc1 - (hc0 - c0 * ec0) * xp.dot(c0, c1)
            for hc1, hc0, c0, ec0, c1 in zip(hci1, _hci0, _ci0, _eci0, ci1_list)
        ]
        hci1 = [
            hc1 - c0 * xp.dot(hc0 - c0 * ec0, c1)
            for hc1, hc0, c0, ec0, c1 in zip(hci1, _hci0, _ci0, _eci0, ci1_list)
        ]

        # Orbital rotation sub-blocks.
        rc = xp.asarray(x1[:, : cache.ncore], dtype=xp.float64) if cache.ncore else xp.zeros((cache.nmo, 0), dtype=xp.float64)
        ra = xp.asarray(x1[:, cache.ncore : cache.nocc], dtype=xp.float64)
        ddm_c = xp.zeros((cache.nmo, cache.nmo), dtype=xp.float64)
        if cache.ncore:
            ddm_c[:, : cache.ncore] = rc[:, : cache.ncore] * 2.0
            ddm_c[: cache.ncore, :] += rc[:, : cache.ncore].T * 2.0

        # Transition RDMs.
        if _HOP_PROFILE:
            if on_gpu: xp.cuda.Stream.null.synchronize()
            _t2_hop = time.perf_counter()
        with _absorb_ctx():
            tdm1, tdm2 = _weighted_trans_rdm12(
                fcisolver,
                ci1_list=ci1_list,
                ci0_list=_ci0,
                weights=_weights,
                ncas=cache.ncas,
                nelecas=cache.nelecas,
                link_index=linkstr,
                return_cupy=on_gpu,
            )
        tdm1 = tdm1 + tdm1.T
        tdm2 = tdm2 + tdm2.transpose(1, 0, 3, 2)
        tdm2 = (tdm2 + tdm2.transpose(2, 3, 0, 1)) * 0.5

        # MO-basis contractions (on whichever device ppaa/papa reside).
        if _HOP_PROFILE:
            if on_gpu: xp.cuda.Stream.null.synchronize()
            _t3_hop = time.perf_counter()
        if cache.ncore:
            if _use_df_factors:
                _naux = int(_L_pu.shape[2])
                _ncas = cache.ncas
                _nmo = cache.nmo
                _ncore = cache.ncore
                _L_uv = _L_pu[_ncore : cache.nocc]  # (ncas, ncas, naux)
                # J: g_Q = sum_{u,v} L_uv[u,v,Q] * tdm1[u,v]
                g_Q = xp.einsum("uvQ,uv->Q", _L_uv, tdm1)
                vhf_a = xp.einsum("piQ,Q->pi", _L_pi, g_Q)
                # K: M[i,u,Q] = sum_v tdm1[u,v] * L_pu[i,v,Q]
                M_iu = xp.einsum("uv,ivQ->iuQ", tdm1, _L_pu[:_ncore])  # (ncore, ncas, naux)
                vhf_a = vhf_a - 0.5 * (
                    _L_pu.reshape(_nmo, _ncas * _naux) @ M_iu.reshape(_ncore, _ncas * _naux).T
                )
            else:
                vhf_a = (
                    xp.einsum("pquv,uv->pq", ppaa[:, : cache.ncore], tdm1, optimize=True)
                    - 0.5 * xp.einsum("puqv,uv->pq", papa[:, :, : cache.ncore], tdm1, optimize=True)
                )
        else:
            vhf_a = xp.empty((cache.nmo, 0), dtype=xp.float64)

        if _use_df_factors:
            _naux = int(_L_pu.shape[2])
            _ncas = cache.ncas
            _nmo = cache.nmo
            _L_uv = _L_pu[cache.ncore : cache.nocc]  # (ncas, ncas, naux)
            # J: g_Q = 2 * sum_{p,i} L_pi[p,i,Q] * ddm_c[p,i]
            if cache.ncore:
                g_Q = 2.0 * xp.einsum("piQ,pi->Q", _L_pi, ddm_c[:, : cache.ncore])
            else:
                g_Q = xp.zeros(_naux, dtype=xp.float64)
            jk = xp.einsum("uvQ,Q->uv", _L_uv, g_Q)
            # K: DL[p,v,Q] = sum_q ddm_c[p,q] * L_pu[q,v,Q]
            DL = (ddm_c @ _L_pu.reshape(_nmo, _ncas * _naux)).reshape(_nmo, _ncas, _naux)
            DL_flat = DL.transpose(1, 0, 2).reshape(_ncas, _nmo * _naux)
            L_pu_flat = _L_pu.transpose(1, 0, 2).reshape(_ncas, _nmo * _naux)
            jk = jk - 0.5 * (L_pu_flat @ DL_flat.T)
        else:
            jk = (
                xp.einsum("pquv,pq->uv", ppaa, ddm_c, optimize=True)
                - 0.5 * xp.einsum("puqv,pq->uv", papa, ddm_c, optimize=True)
            )

        g_dm2 = xp.einsum("puwx,wxuv->pv", paaa, tdm2, optimize=True)

        aaaa = (ra.T @ paaa.reshape(cache.nmo, -1)).reshape((cache.ncas,) * 4)
        aaaa = aaaa + aaaa.transpose(1, 0, 2, 3)
        aaaa = aaaa + aaaa.transpose(2, 3, 0, 1)

        # ── JK for orbital Hessian: MO-basis (fast) or AO-basis (fallback) ──
        if _HOP_PROFILE:
            if on_gpu: xp.cuda.Stream.null.synchronize()
            _t4_hop = time.perf_counter()
        if _use_mo_jk and on_gpu and cache.ncore > 0:
            # MO-basis JK using precomputed L_pq^Q — avoids nao^3 contractions.
            _nmo_L = cache.nmo
            _ncore_L = cache.ncore
            _nocc_L = cache.nocc

            # dm3_MO: core-rest block of x1 (symmetric)
            _dm3 = xp.zeros((_nmo_L, _nmo_L), dtype=xp.float64)
            _dm3[:_ncore_L, _ncore_L:] = x1[:_ncore_L, _ncore_L:]
            _dm3[_ncore_L:, :_ncore_L] = x1[:_ncore_L, _ncore_L:].T
            # dm4_MO: active-all block weighted by casdm1 (symmetric)
            _dm4_h = xp.zeros((_nmo_L, _nmo_L), dtype=xp.float64)
            _dm4_h[_ncore_L:_nocc_L, :] = _casdm1 @ x1[_ncore_L:_nocc_L, :]
            _dm4 = _dm4_h + _dm4_h.T
            _dm_total = _dm3 * 2.0 + _dm4

            _naux_L = int(_L_pq_dev.shape[2])
            # Precomputed views (no copy):
            _L_t_2d = _L_t_dev.reshape(_nmo_L, _naux_L * _nmo_L)  # (nmo, naux*nmo)
            _L_flat = _L_pq_dev.reshape(_nmo_L * _nmo_L, _naux_L)  # (nmo*nmo, naux)
            _L_t_flat = _L_t_2d.reshape(_nmo_L * _naux_L, _nmo_L)  # (nmo*naux, nmo)
            # J0 via GEMV: rho_Q = L_flat^T @ vec(dm3), J0 = reshape(L_flat @ rho)
            _dm3_v = _dm3.ravel()
            _rho0 = _L_flat.T @ _dm3_v
            _J0 = (_L_flat @ _rho0).reshape(_nmo_L, _nmo_L)
            # K0 via two GEMMs (no intermediate copies):
            _LDM0 = (_L_t_flat @ _dm3).reshape(_nmo_L, _naux_L * _nmo_L)
            _K0 = _LDM0 @ _L_t_2d.T
            _v0 = _J0 * 2.0 - _K0
            # J1, K1 for dm_total
            _dmt_v = _dm_total.ravel()
            _rho1 = _L_flat.T @ _dmt_v
            _J1 = (_L_flat @ _rho1).reshape(_nmo_L, _nmo_L)
            _LDM1 = (_L_t_flat @ _dm_total).reshape(_nmo_L, _naux_L * _nmo_L)
            _K1 = _LDM1 @ _L_t_2d.T
            _v1 = _J1 * 2.0 - _K1
            # va = casdm1 @ v0[act, :], vc = v1[:core, rest]
            va = _casdm1 @ _v0[_ncore_L:_nocc_L, :]
            vc = _v1[:_ncore_L, _ncore_L:]
            # Verification removed after confirming machine-precision match.
        elif on_gpu:
            if cache.ncore > 0:
                if _supports_return_gpu:
                    va, vc = casscf.update_jk_in_ah(mo, x1, _casdm1, eris, return_gpu=True)
                else:
                    _va_np, _vc_np = casscf.update_jk_in_ah(mo, x1, _casdm1, eris)
                    va = xp.asarray(_va_np, dtype=xp.float64)
                    vc = xp.asarray(_vc_np, dtype=xp.float64)
        else:
            _jk_on_gpu = False
            if cache.ncore > 0:
                if _supports_return_gpu:
                    va_dev, vc_dev = casscf.update_jk_in_ah(mo, x1, _casdm1, eris, return_gpu=True)
                    _jk_on_gpu = True
                else:
                    va_np, vc_np = casscf.update_jk_in_ah(mo, x1, _casdm1, eris)

        # ── CI Hessian part 2: orbital-CI coupling ──
        if _HOP_PROFILE:
            if on_gpu: xp.cuda.Stream.null.synchronize()
            _t5_hop = time.perf_counter()
        h1aa = (_h1e_mo[cache.ncore : cache.nocc] + _vhf_c[cache.ncore : cache.nocc]) @ ra
        h1aa = h1aa + h1aa.T + jk

        with _ah_mixed_precision_ctx(fcisolver, ah_mixed_precision), _absorb_ctx():
            op_k = fcisolver.absorb_h1e(h1aa, aaaa, cache.ncas, cache.nelecas, 0.5)
            kci0 = [
                fcisolver.contract_2e(op_k, c0, cache.ncas, cache.nelecas, **_c2e_kw).ravel()
                for c0 in _ci0
            ]
        kci0 = [kc0 - xp.dot(kc0, c0) * c0 for kc0, c0 in zip(kci0, _ci0)]
        hci1 = [hc1 + kc0 for hc1, kc0 in zip(hci1, kci0)]
        hci1 = [hc1 * wi for hc1, wi in zip(hci1, _weights)]

        # ── Orbital Hessian assembly ──
        if _HOP_PROFILE:
            if on_gpu: xp.cuda.Stream.null.synchronize()
            _t6_hop = time.perf_counter()
        x2 = (_h1e_mo @ x1) @ _dm1_full
        g_orb_mat = xp.einsum("r,rpq->pq", _weights, _gpq, optimize=True)
        x2 -= (g_orb_mat + g_orb_mat.T) @ x1 * 0.5
        if cache.ncore:
            x2[: cache.ncore] += (x1[: cache.ncore, cache.ncore :] @ _vhf_ca[cache.ncore :]) * 2.0
        x2[cache.ncore : cache.nocc] += (_casdm1 @ x1[cache.ncore : cache.nocc]) @ _vhf_c
        x2[:, cache.ncore : cache.nocc] += xp.einsum("purv,rv->pu", _hdm2, x1[:, cache.ncore : cache.nocc], optimize=True)

        # ── JK sync ──
        if on_gpu:
            if cache.ncore > 0:
                x2[cache.ncore : cache.nocc] += va
                x2[: cache.ncore, cache.ncore :] += vc
        else:
            if cache.ncore > 0:
                if _jk_on_gpu:
                    va = _to_np_f64(va_dev)
                    vc = _to_np_f64(vc_dev)
                else:
                    va, vc = va_np, vc_np
                x2[cache.ncore : cache.nocc] += va
                x2[: cache.ncore, cache.ncore :] += vc

        # SA overlap contribution.
        s10 = xp.asarray(
            [xp.dot(c1, c0) * 2.0 * wi for c1, c0, wi in zip(ci1_list, _ci0, _weights)],
            dtype=xp.float64,
        )
        if cache.ncore:
            x2[:, : cache.ncore] += ((_h1e_mo[:, : cache.ncore] + _vhf_c[:, : cache.ncore]) * xp.sum(s10) + vhf_a) * 2.0
        x2[:, cache.ncore : cache.nocc] += (_h1e_mo[:, cache.ncore : cache.nocc] + _vhf_c[:, cache.ncore : cache.nocc]) @ tdm1
        x2[:, cache.ncore : cache.nocc] += g_dm2
        x2 -= xp.einsum("r,rpq->pq", s10, _gpq, optimize=True)
        x2 = x2 - x2.T

        # Pack output.
        packed_orb = casscf.pack_uniq_var(x2) * 2.0  # xp-aware (Step 5)
        if on_gpu:
            if nroots == 1:
                packed_ci = hci1[0].ravel() * 2.0
            else:
                packed_ci = xp.concatenate([v.ravel() for v in hci1]) * 2.0
            out = xp.concatenate([packed_orb.ravel(), packed_ci])
        else:
            out = np.hstack((packed_orb, cache.ci.pack(hci1) * 2.0))
        out = xp.asarray(out, dtype=xp.float64).ravel()
        # Davidson solver operates in NumPy — convert GPU result back.
        if on_gpu and _hop_on_gpu:
            out = out.get()
        if _HOP_PROFILE:
            _t7_hop = time.perf_counter()
            print(
                f"  h_op profile: "
                f"unpack={(_t1_hop-_t0_hop)*1e3:.1f}ms  "
                f"sigma={(_t2_hop-_t1_hop)*1e3:.1f}ms  "
                f"trdm={(_t3_hop-_t2_hop)*1e3:.1f}ms  "
                f"mo_contr={(_t4_hop-_t3_hop)*1e3:.1f}ms  "
                f"jk={'MO' if _use_mo_jk else 'AO'}={(_t5_hop-_t4_hop)*1e3:.1f}ms  "
                f"ci_coup={(_t6_hop-_t5_hop)*1e3:.1f}ms  "
                f"orb_hess={(_t7_hop-_t6_hop)*1e3:.1f}ms  "
                f"total={(_t7_hop-_t0_hop)*1e3:.1f}ms",
                flush=True,
            )
        return out

    gauge_l = str(gauge).strip().lower()
    if gauge_l not in ("none", "project", "project_out"):
        raise ValueError("gauge must be one of: none, project, project_out")
    if nroots > 1 and gauge_l != "none":
        n_ci_total = int(sum(int(np.asarray(c).size) for c in ci0_list))
        n_orb = int(g_all.size) - n_ci_total
        h_op = _wrap_h_op_ci_projection(
            _h_op_raw,
            n_orb=n_orb,
            ci_ref_list=ci0_list,
            project_input=(gauge_l == "project"),
            project_output=True,
        )
    else:
        h_op = _h_op_raw

    return g_all, g_update, h_op, cache.hdiag_all


def gen_g_hop(
    casscf: Any,
    mo: np.ndarray,
    ci0: Any,
    eris: Any,
    verbose: int | None = None,
    *,
    weights: Sequence[float] | None = None,
    gauge: str = "none",
    convention: str = "pyscf2",
    strict_weights: bool = False,
    enforce_absorb_h1e_direct: bool = True,
    implementation: str = "internal",
    ah_mixed_precision: bool = False,
) -> tuple[np.ndarray, Callable[..., Any], Callable[[np.ndarray], np.ndarray], Any]:
    """Return (g_all, g_update, h_op, h_diag) with a standard Newton-CASSCF surface.

    Parameters
    ----------
    casscf : Any
        CASSCF object.
    mo : np.ndarray
        Molecular orbitals.
    ci0 : Any
        Reference CI vector(s).
    eris : Any
        Integral object.
    verbose : int | None, optional
        Verbosity level.
    weights : Sequence[float] | None, optional
        State-average weights.
    gauge : str, optional
        Gauge regarding CI projection ("none", "project_out", "project").
    convention : str, optional
        Conventions version ("pyscf2").
    strict_weights : bool, optional
        Whether to enforce weight consistency.
    enforce_absorb_h1e_direct : bool, optional
        Whether to force direct H1e absorption.
    implementation : str, optional
        Implementation backend ("internal").

    Returns
    -------
    tuple
        (g, g_update, h_op, h_diag)
    """

    convention_l = str(convention).strip().lower()
    if convention_l not in ("pyscf2",):
        raise ValueError("unsupported convention (Phase 0 supports convention='pyscf2' only)")

    impl = str(implementation).strip().lower()
    if impl not in ("internal", "cuguga", "owned"):
        raise ValueError("implementation must be one of: internal")
    return gen_g_hop_internal(
        casscf,
        mo,
        ci0,
        eris,
        verbose=verbose,
        weights=weights,
        gauge=gauge,
        convention=convention,
        strict_weights=bool(strict_weights),
        enforce_absorb_h1e_direct=bool(enforce_absorb_h1e_direct),
        ah_mixed_precision=bool(ah_mixed_precision),
    )


@dataclass
class NewtonMicroStats:
    """Micro-iteration stats returned by `update_orb_ci`."""

    imic: int = 0
    tot_hop: int = 0
    tot_kf: int = 0

    # Diagnostics / instrumentation (best-effort; not used in core control flow).
    n_trust_fail_grad: int = 0
    n_trust_fail_kf: int = 0
    n_retry: int = 0
    n_mu_increase: int = 0
    n_step_scaled: int = 0
    n_step_scaled_orb: int = 0
    n_step_scaled_ci: int = 0
    last_level_shift: float = 0.0
    last_mu_orb: float = 0.0
    last_mu_ci: float = 0.0
    last_kf_trust: float = 0.0
    g_orb: np.ndarray | None = None  # orbital gradient at keyframe entry


def _orthonormalize_ci_columns(
    ci_list: Sequence[np.ndarray],
    *,
    ref_list: Sequence[np.ndarray] | None = None,
    eps: float = 1e-12,
) -> list[np.ndarray]:
    """Symmetric orthonormalization for CI root columns (small nroots).

    Parameters
    ----------
    ci_list : Sequence[np.ndarray]
        List of CI vectors to orthonormalize.
    ref_list : Sequence[np.ndarray] | None, optional
        Reference CI vectors for phase alignment.
    eps : float, optional
        Small epsilon for eigenvalues.

    Returns
    -------
    list[np.ndarray]
        Orthonormalized CI vectors.
    """

    c_list = [np.asarray(c, dtype=np.float64).ravel() for c in ci_list]
    nroots = int(len(c_list))
    if nroots == 0:
        raise ValueError("empty CI list")
    if nroots == 1:
        c0 = c_list[0]
        nrm = float(np.linalg.norm(c0))
        return [c0 / nrm] if nrm > 0.0 else [c0]

    nci = int(c_list[0].size)
    if any(int(c.size) != nci for c in c_list):
        raise ValueError("inconsistent CI sizes across roots")

    cmat = np.stack(c_list, axis=1)  # (nci,nroots)
    s = cmat.T @ cmat
    evals, evecs = np.linalg.eigh(s)
    evals = np.maximum(evals, float(eps))
    s_inv_sqrt = (evecs * (1.0 / np.sqrt(evals))[None, :]) @ evecs.T
    q = cmat @ s_inv_sqrt

    if ref_list is not None:
        ref = [np.asarray(c, dtype=np.float64).ravel() for c in ref_list]
        if len(ref) == nroots and all(int(r.size) == nci for r in ref):
            for i in range(nroots):
                if float(np.dot(ref[i], q[:, i])) < 0.0:
                    q[:, i] *= -1.0

    return [np.ascontiguousarray(q[:, i]) for i in range(nroots)]


def extract_rotation(
    casscf: Any,
    dr: np.ndarray,
    u: np.ndarray,
    ci0: Any,
    *,
    ci_update: str = "pyscf",
) -> tuple[np.ndarray, Any]:
    """Apply a packed step `dr` to (u, ci0) and return updated (u, ci1).

    This mirrors PySCF's `pyscf.mcscf.newton_casscf.extract_rotation`, but adds an
    optional multi-root orthonormalization step (`ci_update="orthonormalize"`).

    Parameters
    ----------
    casscf : Any
        CASSCF object.
    dr : np.ndarray
        Packed update vector (orbital rotation + CI update).
    u : np.ndarray
        Current orbital rotation matrix.
    ci0 : Any
        Current CI vector(s).
    ci_update : str, optional
        CI update method ("pyscf" or "orthonormalize").

    Returns
    -------
    tuple
        (u_new, ci_new)
    """

    dr = np.asarray(dr, dtype=np.float64).ravel()
    u = np.asarray(u, dtype=np.float64)

    nmo = int(casscf.mo_coeff.shape[1])
    frozen = getattr(casscf, "frozen", None)
    ngorb = int(np.count_nonzero(casscf.uniq_var_indices(nmo, casscf.ncore, casscf.ncas, frozen)))

    u = u @ casscf.update_rotate_matrix(dr[:ngorb])

    ci0_list = _as_ci_list(ci0)
    p0 = int(ngorb)
    ci1_list: list[np.ndarray] = []
    for c0 in ci0_list:
        p1 = p0 + int(c0.size)
        d = np.asarray(c0, dtype=np.float64).ravel() + dr[p0:p1]
        nrm = float(np.linalg.norm(d))
        if nrm > 0.0:
            d = d / nrm
        ci1_list.append(d)
        p0 = p1

    if len(ci1_list) > 1:
        mode = str(ci_update).strip().lower()
        if mode not in ("pyscf", "orthonormalize"):
            raise ValueError("ci_update must be 'pyscf' or 'orthonormalize'")
        if mode == "orthonormalize":
            ci1_list = _orthonormalize_ci_columns(ci1_list, ref_list=ci0_list)

    if isinstance(ci0, np.ndarray):
        return u, ci1_list[0]
    return u, ci1_list


def update_orb_ci(
    casscf: Any,
    mo: np.ndarray,
    ci0: Any,
    eris: Any,
    x0_guess: np.ndarray | None = None,
    conv_tol_grad: float = 1e-4,
    max_stepsize: float | None = None,
    verbose: int | None = None,
    *,
    weights: Sequence[float] | None = None,
    gauge: str = "none",
    convention: str = "pyscf2",
    strict_weights: bool = False,
    enforce_absorb_h1e_direct: bool = True,
    implementation: str = "internal",
    ci_update: str = "pyscf",
    ah_mixed_precision: bool = False,
) -> tuple[np.ndarray, Any, float, NewtonMicroStats, np.ndarray]:
    """Newton/AH micro-iterations updating orbitals+CI using `gen_g_hop`.

    This is a cuGUGA-owned port of PySCF's `pyscf.mcscf.newton_casscf.update_orb_ci`,
    parameterized by `gen_g_hop(..., implementation=...)` so the same operator is
    used in both operator-only and full-optimizer modes.

    Parameters
    ----------
    casscf : Any
        CASSCF object.
    mo : np.ndarray
        Molecular orbitals.
    ci0 : Any
        Reference CI vector(s).
    eris : Any
        Integral object.
    x0_guess : np.ndarray | None, optional
        Initial guess for the update step.
    conv_tol_grad : float, optional
        Gradient convergence tolerance.
    max_stepsize : float | None, optional
        Maximum step size.
    verbose : int | None, optional
        Verbosity level.
    weights : Sequence[float] | None, optional
        State-average weights.
    gauge : str, optional
        Gauge regarding CI projection.
    convention : str, optional
        Conventions version.
    strict_weights : bool, optional
        Whether to enforce weight consistency.
    enforce_absorb_h1e_direct : bool, optional
        Whether to force direct H1e absorption.
    implementation : str, optional
        Implementation backend.
    ci_update : str, optional
        CI update method.

    Returns
    -------
    tuple
        (u_new, ci_new, norm_g_kf, stats, last_dxi)
    """

    log = _new_logger(casscf, verbose)
    verbose_level = int(getattr(log, "verbose", getattr(casscf, "verbose", 0)))
    if max_stepsize is None:
        max_stepsize = float(getattr(casscf, "max_stepsize", 0.03))

    nmo = int(mo.shape[1])

    ci0_list = _as_ci_list(ci0)
    ci0_use: Any = ci0_list[0] if len(ci0_list) == 1 and isinstance(ci0, np.ndarray) else ci0_list

    g_all, g_update, h_op, h_diag = gen_g_hop(
        casscf,
        mo,
        ci0_use,
        eris,
        verbose=verbose_level,
        weights=weights,
        gauge=gauge,
        convention=convention,
        strict_weights=bool(strict_weights),
        enforce_absorb_h1e_direct=bool(enforce_absorb_h1e_direct),
        implementation=implementation,
        ah_mixed_precision=bool(ah_mixed_precision),
    )
    g_all = np.asarray(g_all, dtype=np.float64).ravel()

    frozen = getattr(casscf, "frozen", None)
    ngorb = int(np.count_nonzero(casscf.uniq_var_indices(nmo, casscf.ncore, casscf.ncas, frozen)))

    norm_gkf = norm_gall = float(np.linalg.norm(g_all))
    log.debug(
        "    |g|=%5.3g (%4.3g %4.3g) (keyframe)",
        norm_gall,
        float(np.linalg.norm(g_all[:ngorb])),
        float(np.linalg.norm(g_all[ngorb:])),
    )

    # ── Local knobs (AH inner-loop only) ─────────────────────────────────────
    # These attributes are intentionally read via getattr so callers can
    # override them on the adapter object without changing public APIs.
    root_v0_min = float(getattr(casscf, "ah_root_v0_min", 0.1))
    root_homing = bool(getattr(casscf, "ah_root_homing_enabled", True))
    root_pred = bool(getattr(casscf, "ah_root_pred_decrease_enabled", True))
    root_pred_tol = float(getattr(casscf, "ah_root_pred_decrease_tol_rel", 1e-3))

    start_tol_base = float(getattr(casscf, "ah_start_tol", 2.5))
    start_tol_dynamic = bool(getattr(casscf, "ah_start_tol_dynamic", True))

    retry_enabled = bool(getattr(casscf, "ah_retry_enabled", True))
    retry_max = int(getattr(casscf, "ah_retry_max", 6))
    level_shift_growth = float(getattr(casscf, "ah_level_shift_growth", 5.0))
    stepsize_shrink = float(getattr(casscf, "ah_stepsize_shrink", 0.5))

    mu_enabled = bool(getattr(casscf, "ah_mu_enabled", True))
    mu_init = float(getattr(casscf, "ah_mu_init", 0.0))
    mu_growth = float(getattr(casscf, "ah_mu_growth", 5.0))
    mu_max = float(getattr(casscf, "ah_mu_max", 1e6))
    mu_blockwise = bool(getattr(casscf, "ah_mu_blockwise", True))
    mu_ci_scale = float(getattr(casscf, "ah_mu_ci_scale", 1.0))
    scale_fallback = bool(getattr(casscf, "ah_scale_fallback", True))

    mgs = bool(getattr(casscf, "ah_davidson_mgs_enabled", True))
    d_restart = bool(getattr(casscf, "ah_davidson_restart_enabled", True))
    d_restart_stagnant = int(getattr(casscf, "ah_davidson_restart_stagnant", 3))

    kf_min_ikf = int(getattr(casscf, "ah_keyframe_min_ikf", 2))
    kf_trust_dynamic = bool(getattr(casscf, "ah_kf_trust_dynamic", True))
    kf_trust_min = float(getattr(casscf, "ah_kf_trust_min", 1.2))
    kf_trust_max = float(getattr(casscf, "ah_kf_trust_max", 10.0))

    # Separate caps for orbital and CI blocks (same vector scaling factor).
    max_stepsize_orb = float(getattr(casscf, "ah_max_stepsize_orb", max_stepsize))
    max_stepsize_ci = float(
        getattr(
            casscf,
            "ah_max_stepsize_ci",
            max(0.1, min(0.5, max_stepsize_orb * 10.0)),
        )
    )

    level_shift_local = float(getattr(casscf, "ah_level_shift_local_init", casscf.ah_level_shift))
    mu_orb = float(mu_init)
    mu_ci = float(mu_init * mu_ci_scale) if mu_blockwise else float(mu_init)

    def _step_maxabs(x: np.ndarray) -> tuple[float, float, float]:
        x = np.asarray(x, dtype=np.float64).ravel()
        max_all = float(np.max(np.abs(x))) if x.size else 0.0
        max_orb = float(np.max(np.abs(x[:ngorb]))) if ngorb > 0 else 0.0
        max_ci = float(np.max(np.abs(x[ngorb:]))) if ngorb < int(x.size) else 0.0
        return max_orb, max_ci, max_all

    def _scale_to_caps(dxi: np.ndarray, hdxi: np.ndarray, cap_orb: float, cap_ci: float) -> tuple[np.ndarray, np.ndarray]:
        max_orb, max_ci, _ = _step_maxabs(dxi)
        scale = 1.0
        if max_orb > cap_orb:
            scale = min(scale, cap_orb / max_orb)
        if max_ci > cap_ci:
            scale = min(scale, cap_ci / max_ci)
        if scale < 1.0:
            stat.n_step_scaled += 1
            if max_orb > cap_orb:
                stat.n_step_scaled_orb += 1
            if max_ci > cap_ci:
                stat.n_step_scaled_ci += 1
            log.debug1("Scale AH step by %g (caps orb=%g ci=%g)", scale, cap_orb, cap_ci)
            dxi = np.asarray(dxi, dtype=np.float64) * scale
            hdxi = np.asarray(hdxi, dtype=np.float64) * scale
        return dxi, hdxi

    def _mu_correct_hdxi(hdxi: np.ndarray, dxi: np.ndarray, muo: float, muc: float) -> np.ndarray:
        """Convert (H+mu)x to Hx without extra h_op calls."""
        hdxi = np.asarray(hdxi, dtype=np.float64).ravel().copy()
        dxi = np.asarray(dxi, dtype=np.float64).ravel()
        if muo != 0.0 and ngorb > 0:
            hdxi[:ngorb] -= float(muo) * dxi[:ngorb]
        if muc != 0.0 and ngorb < int(dxi.size):
            hdxi[ngorb:] -= float(muc) * dxi[ngorb:]
        return hdxi

    def _make_precond(hdiag: Any, level_shift: float, muo: float, muc: float) -> Callable[[np.ndarray, float], np.ndarray]:
        def _p(x: np.ndarray, e: float) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64).ravel()
            if callable(hdiag):
                # Callable preconditioners are assumed to already handle sign/shift robustly.
                out = np.asarray(hdiag(x, e - float(level_shift)), dtype=np.float64).ravel()
                nrm = float(np.linalg.norm(out))
                if nrm > 0.0:
                    out *= 1.0 / nrm
                return out

            hdiagd = np.asarray(hdiag, dtype=np.float64).ravel() - (float(e) - float(level_shift))
            if muo != 0.0 and ngorb > 0:
                hdiagd[:ngorb] += float(muo)
            if muc != 0.0 and ngorb < int(hdiagd.size):
                hdiagd[ngorb:] += float(muc)

            # Sign-preserving floor for near-singular denominators.
            eps = 1e-8
            mask = np.abs(hdiagd) < eps
            if np.any(mask):
                hdiagd = hdiagd.copy()
                hdiagd[mask] = np.copysign(eps, hdiagd[mask])
            out = x / hdiagd
            nrm = float(np.linalg.norm(out))
            if nrm > 0.0:
                out *= 1.0 / nrm
            return out

        return _p

    def _make_h_op_mu(hop: Callable[[np.ndarray], np.ndarray], muo: float, muc: float) -> Callable[[np.ndarray], np.ndarray]:
        if muo == 0.0 and muc == 0.0:
            return hop

        def _hop_mu(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64).ravel()
            y = np.asarray(hop(x), dtype=np.float64).ravel()
            if muo != 0.0 and ngorb > 0:
                y[:ngorb] += float(muo) * x[:ngorb]
            if muc != 0.0 and ngorb < int(x.size):
                y[ngorb:] += float(muc) * x[ngorb:]
            return y

        return _hop_mu

    stat = NewtonMicroStats(imic=0, tot_hop=0, tot_kf=1)
    stat.g_orb = g_all[:ngorb].copy() if ngorb > 0 else None
    dr = np.zeros_like(g_all)
    ikf = 0
    u = np.eye(nmo, dtype=np.float64)
    ci_kf: Any = ci0_use
    kf_trust_local = float(getattr(casscf, "kf_trust_region", 3.0))

    if x0_guess is None:
        x0_guess = g_all
    x0_guess = np.asarray(x0_guess, dtype=np.float64).ravel()
    g_op = lambda: g_all

    if norm_gall < float(conv_tol_grad) * 0.3:
        return u, ci_kf, norm_gall, stat, x0_guess

    last_dxi = np.asarray(x0_guess, dtype=np.float64).ravel()
    last_hdxi_true = np.zeros_like(last_dxi)

    hop_total = 0
    n_retry = 0
    max_cycle_cur = int(casscf.max_cycle_micro)
    while True:
        precond = _make_precond(h_diag, level_shift_local, mu_orb, mu_ci)
        h_op_use = _make_h_op_mu(h_op, mu_orb, mu_ci)

        stat.last_level_shift = float(level_shift_local)
        stat.last_mu_orb = float(mu_orb)
        stat.last_mu_ci = float(mu_ci)
        stat.last_kf_trust = float(kf_trust_local)

        x0_use = np.asarray(g_all, dtype=np.float64).ravel() if n_retry > 0 else np.asarray(last_dxi, dtype=np.float64).ravel()
        hop_before = hop_total
        restart = False
        terminate = False

        for ah_conv, ihop, w, dxi, hdxi, residual, seig in davidson_cc(
            h_op_use,
            g_op,
            precond,
            x0_use,
            tol=casscf.ah_conv_tol,
            max_cycle=casscf.ah_max_cycle,
            lindep=casscf.ah_lindep,
            verbose=log,
            root_v0_min=root_v0_min,
            root_homing=root_homing,
            root_pred_decrease=root_pred,
            root_pred_decrease_tol_rel=root_pred_tol,
            trust_maxabs_orb=max_stepsize_orb,
            trust_maxabs_ci=max_stepsize_ci,
            ngorb=ngorb,
            mu_orb=mu_orb,
            mu_ci=mu_ci,
            mgs=mgs,
            restart=d_restart,
            restart_stagnant=d_restart_stagnant,
        ):
            hop_total = hop_before + int(ihop)
            stat.tot_hop = int(hop_total)

            norm_residual = float(np.linalg.norm(residual))
            start_tol_eff = float(start_tol_base)
            if start_tol_dynamic:
                start_tol_eff = min(5.0 * float(norm_gall), float(start_tol_base))

            accept_step = (
                bool(ah_conv)
                or int(ihop) == int(casscf.ah_max_cycle)
                or ((norm_residual < float(start_tol_eff)) and (int(ihop) >= int(casscf.ah_start_cycle)))
                or (float(seig) < float(casscf.ah_lindep))
            )
            if not accept_step:
                continue

            dxi = np.asarray(dxi, dtype=np.float64).ravel()
            hdxi = np.asarray(hdxi, dtype=np.float64).ravel()
            hdxi_true = _mu_correct_hdxi(hdxi, dxi, mu_orb, mu_ci)

            # Enforce step caps by adjusting mu (preferred) or scaling (fallback).
            max_orb, max_ci, max_all = _step_maxabs(dxi)
            if (max_orb > max_stepsize_orb) or (max_ci > max_stepsize_ci):
                if mu_enabled and (mu_orb < mu_max or mu_ci < mu_max) and n_retry < retry_max:
                    if not mu_blockwise:
                        mu_orb = min(float(mu_max), max(float(mu_orb), 1e-12) * float(mu_growth))
                        mu_ci = mu_orb
                    else:
                        if max_orb > max_stepsize_orb:
                            mu_orb = min(float(mu_max), max(float(mu_orb), 1e-12) * float(mu_growth))
                        if max_ci > max_stepsize_ci:
                            mu_ci = min(float(mu_max), max(float(mu_ci), 1e-12) * float(mu_growth))
                    stat.n_mu_increase += 1
                    stat.n_retry += 1
                    n_retry += 1
                    restart = True
                    log.debug("AH step exceeds caps (orb=%g ci=%g); increase mu and retry", max_orb, max_ci)
                    break
                if scale_fallback:
                    dxi, hdxi_true = _scale_to_caps(dxi, hdxi_true, max_stepsize_orb, max_stepsize_ci)
                    max_orb, max_ci, max_all = _step_maxabs(dxi)

            # Predict gradient after the step; reject early on trust-region violation.
            g_trial = np.asarray(g_all, dtype=np.float64).ravel() + np.asarray(hdxi_true, dtype=np.float64).ravel()
            norm_g_trial = float(np.linalg.norm(g_trial))
            if stat.imic >= 3 and norm_g_trial > norm_gkf * float(casscf.ah_grad_trust_region):
                stat.n_trust_fail_grad += 1
                stat.n_retry += 1
                n_retry += 1
                if retry_enabled and n_retry <= retry_max:
                    level_shift_local = max(float(level_shift_local), 1e-16) * float(level_shift_growth)
                    max_stepsize_orb = max(float(max_stepsize_orb) * float(stepsize_shrink), 1e-6)
                    max_stepsize_ci = max(float(max_stepsize_ci) * float(stepsize_shrink), 1e-6)
                    if mu_enabled and (mu_orb < mu_max or mu_ci < mu_max):
                        mu_orb = min(float(mu_max), max(float(mu_orb), 1e-12) * float(mu_growth))
                        mu_ci = min(float(mu_max), max(float(mu_ci), 1e-12) * float(mu_growth)) if mu_blockwise else mu_orb
                        stat.n_mu_increase += 1
                    restart = True
                    log.debug("|g| >> keyframe (trial). Damp+retry (level_shift=%g)", level_shift_local)
                    break
                log.debug("|g| >> keyframe (trial). Stop inner iterations (retry disabled/exhausted)")
                terminate = True
                break

            # Commit the micro step.
            stat.imic += 1
            last_dxi = dxi
            last_hdxi_true = hdxi_true
            dr += dxi
            g_all = g_trial
            n_retry = 0
            norm_dr = float(np.linalg.norm(dr))
            norm_gall = float(np.linalg.norm(g_all))
            norm_gorb = float(np.linalg.norm(g_all[:ngorb]))
            norm_gci = float(np.linalg.norm(g_all[ngorb:]))
            log.debug(
                "    imic %d(%d)  |g|=%3.2e (%2.1e %2.1e)  |dxi|=%3.2e max(o,c,a)=(%3.2e %3.2e %3.2e) |dr|=%3.2e  eig=%2.1e seig=%2.1e",
                stat.imic,
                hop_total,
                norm_gall,
                norm_gorb,
                norm_gci,
                float(np.linalg.norm(dxi)),
                max_orb,
                max_ci,
                max_all,
                norm_dr,
                float(w),
                float(seig),
            )

            max_cycle_cur = max(
                int(casscf.max_cycle_micro),
                int(casscf.max_cycle_micro) - int(np.log(norm_gkf + 1e-7) * 2),
            )
            max_cycle_cap = getattr(casscf, "ah_max_cycle_micro_cap", None)
            if max_cycle_cap is not None:
                max_cycle_cur = min(max_cycle_cur, max(int(casscf.max_cycle_micro), int(max_cycle_cap)))
            log.debug1("Set max_cycle %d", max_cycle_cur)
            ikf += 1

            if stat.imic >= int(max_cycle_cur) or norm_gall < float(conv_tol_grad) * 0.3:
                break

            # Keyframe update: recompute gradient (and refresh h_op/h_diag) in a trial state,
            # then accept/reject without mutating the current keyframe unless accepted.
            kf_trigger = (
                ikf >= max(
                    int(casscf.kf_interval),
                    int(casscf.kf_interval) - int(np.log(norm_dr + 1e-7)),
                )
                or (norm_gall < norm_gkf / float(kf_trust_local))
            )
            if kf_trigger and ikf > int(kf_min_ikf):
                u_trial, ci_trial = extract_rotation(casscf, dr, u, ci_kf, ci_update=ci_update)
                g_kf1, h_op_kf1, h_diag_kf1 = g_update(u_trial, ci_trial)
                g_kf1 = np.asarray(g_kf1, dtype=np.float64).ravel()
                stat.tot_kf += 1

                norm_gkf1 = float(np.linalg.norm(g_kf1))
                norm_gorb = float(np.linalg.norm(g_kf1[:ngorb]))
                norm_gci = float(np.linalg.norm(g_kf1[ngorb:]))
                norm_dg = float(np.linalg.norm(g_kf1 - g_all))
                log.debug(
                    "Adjust keyframe to |g|= %4.3g (%4.3g %4.3g) |g-correction|= %4.3g",
                    norm_gkf1,
                    norm_gorb,
                    norm_gci,
                    norm_dg,
                )

                # Dynamic keyframe trust region (CIAH-style).
                if kf_trust_dynamic:
                    ratio = float(norm_gall) / (float(norm_dg) + 1e-12)
                    kf_trust_local = float(np.clip(ratio, float(kf_trust_min), float(kf_trust_max)))
                    stat.last_kf_trust = float(kf_trust_local)

                if (
                    norm_dg < norm_gall * float(casscf.ah_grad_trust_region)
                    or norm_gkf1 < float(conv_tol_grad) * float(casscf.ah_grad_trust_region)
                ):
                    u, ci_kf = u_trial, ci_trial
                    dr[:] = 0.0
                    g_all = g_kf1
                    h_op = h_op_kf1
                    h_diag = h_diag_kf1
                    norm_gall = norm_gkf = norm_gkf1
                    ikf = 0
                    n_retry = 0
                    # Restart Davidson at the new keyframe/operator.
                    restart = True
                    break

                # Out of trust region: undo the last micro step and damp+retry.
                stat.n_trust_fail_kf += 1
                stat.n_retry += 1
                n_retry += 1
                dr -= last_dxi
                g_all = g_all - last_hdxi_true
                stat.imic = max(int(stat.imic) - 1, 0)
                ikf = max(int(ikf) - 1, 0)
                norm_gall = float(np.linalg.norm(g_all))
                norm_gkf = float(norm_gkf)
                log.debug("Out of trust region. Undo last step and damp+retry")

                if retry_enabled and n_retry <= retry_max:
                    level_shift_local = max(float(level_shift_local), 1e-16) * float(level_shift_growth)
                    max_stepsize_orb = max(float(max_stepsize_orb) * float(stepsize_shrink), 1e-6)
                    max_stepsize_ci = max(float(max_stepsize_ci) * float(stepsize_shrink), 1e-6)
                    if mu_enabled and (mu_orb < mu_max or mu_ci < mu_max):
                        mu_orb = min(float(mu_max), max(float(mu_orb), 1e-12) * float(mu_growth))
                        mu_ci = min(float(mu_max), max(float(mu_ci), 1e-12) * float(mu_growth)) if mu_blockwise else mu_orb
                        stat.n_mu_increase += 1
                    restart = True
                    break

                # Give up if retry is disabled or exhausted.
                restart = False
                break

        # End davidson loop
        if terminate:
            break
        if restart:
            # Restart inner AH solve with updated (mu, level_shift, caps, operator).
            if n_retry > retry_max:
                break
            continue
        break

    u, ci_kf = extract_rotation(casscf, dr, u, ci_kf, ci_update=ci_update)
    try:
        if isinstance(ci_kf, list) and isinstance(ci0_use, list):
            dci_kf = np.concatenate([(np.asarray(x) - np.asarray(y)).ravel() for x, y in zip(ci_kf, ci0_use)])
        else:
            dci_kf = np.asarray(ci_kf, dtype=np.float64).ravel() - np.asarray(ci0_use, dtype=np.float64).ravel()
    except Exception:
        dci_kf = np.zeros(1, dtype=np.float64)
    log.debug(
        "    tot inner=%d  |g|= %4.3g (%4.3g %4.3g) |u-1|= %4.3g  |dci|= %4.3g",
        stat.imic,
        norm_gall,
        float(np.linalg.norm(g_all[:ngorb])),
        float(np.linalg.norm(g_all[ngorb:])),
        float(np.linalg.norm(u - np.eye(nmo))),
        float(np.linalg.norm(dci_kf)),
    )

    return u, ci_kf, norm_gkf, stat, np.asarray(last_dxi, dtype=np.float64).ravel()


def kernel_newton(
    casscf: Any,
    mo_coeff: np.ndarray,
    tol: float = 1e-7,
    conv_tol_grad: float | None = None,
    ci0: Any | None = None,
    callback: Callable[[dict[str, Any]], Any] | None = None,
    verbose: int | None = None,
    dump_chk: bool = True,
    *,
    weights: Sequence[float] | None = None,
    gauge: str = "none",
    convention: str = "pyscf2",
    strict_weights: bool = False,
    enforce_absorb_h1e_direct: bool = True,
    implementation: str = "internal",
    ci_update: str = "pyscf",
    ah_mixed_precision: bool = False,
) -> tuple[bool, float, Any, Any, np.ndarray, Any]:
    """Second-order (Newton/AH) CASSCF driver using cuGUGA `gen_g_hop`.

    This is a cuGUGA-owned port of PySCF's `pyscf.mcscf.newton_casscf.kernel`
    that routes Hessian/gradient evaluation through this module's `gen_g_hop`.

    Parameters
    ----------
    casscf : Any
        CASSCF object.
    mo_coeff : np.ndarray
        Initial molecular orbitals.
    tol : float, optional
        Energy convergence tolerance.
    conv_tol_grad : float | None, optional
        Gradient convergence tolerance.
    ci0 : Any | None, optional
        Initial CI vector(s).
    callback : Callable | None, optional
        Callback function.
    verbose : int | None, optional
        Verbosity level.
    dump_chk : bool, optional
        Whether to dump checkpoints.
    weights : Sequence[float] | None, optional
        State-average weights.
    gauge : str, optional
        Gauge regarding CI projection.
    convention : str, optional
        Conventions version.
    strict_weights : bool, optional
        Whether to enforce weight consistency.
    enforce_absorb_h1e_direct : bool, optional
        Whether to force direct H1e absorption.
    implementation : str, optional
        Implementation backend.
    ci_update : str, optional
        CI update method.

    Returns
    -------
    tuple
        (converged, e_tot, e_cas, fcivec, mo, mo_energy)
    """

    log = _new_logger(casscf, verbose)
    cput0 = (time.process_time(), time.perf_counter())
    log.debug("Start cuGUGA newton CASSCF")

    if callback is None:
        callback = getattr(casscf, "callback", None)
    if ci0 is None:
        ci0 = getattr(casscf, "ci", None)

    # Ensure SA-CASSCF weights are consistent for the entire optimization.
    # (PySCF scatters weights across mc.weights and mc.fcisolver.weights.)
    weights_use = weights
    ctx_w_mc = nullcontext(False)
    ctx_w_fs = nullcontext(False)
    if ci0 is not None:
        nroots = int(len(_as_ci_list(ci0)))
    else:
        fs_guess = getattr(casscf, "fcisolver", None)
        nroots = int(getattr(fs_guess, "nroots", 1) or 1)
    if nroots > 1:
        w_info = _resolve_weights(casscf, nroots=nroots, weights=weights, strict=bool(strict_weights))
        w_list = [float(x) for x in w_info.weights.tolist()]
        weights_use = w_list
        fcisolver = getattr(casscf, "fcisolver", None)
        ctx_w_mc = _maybe_set_attr(casscf, "weights", w_list)
        ctx_w_fs = _maybe_set_attr(fcisolver, "weights", w_list)

    mo = mo_coeff
    nmo = int(mo.shape[1])

    with ctx_w_mc, ctx_w_fs:
        eris = casscf.ao2mo(mo)
        e_tot, e_cas, fcivec = casscf.casci(mo, ci0, eris, log, locals())
        if casscf.ncas == nmo and not getattr(casscf, "internal_rotation", False):
            if getattr(casscf, "canonicalization", False):
                log.debug("CASSCF canonicalization")
                mo, fcivec, mo_energy = casscf.canonicalize(
                    mo,
                    fcivec,
                    eris,
                    casscf.sorting_mo_energy,
                    casscf.natorb,
                    verbose=log,
                )
            else:
                mo_energy = None
            return True, float(e_tot), e_cas, fcivec, mo, mo_energy

        if conv_tol_grad is None:
            conv_tol_grad = float(np.sqrt(float(tol)))
            log.info("Set conv_tol_grad to %g", conv_tol_grad)

        conv = False
        de = elast = float(e_tot)
        dr0 = None
        imacro = 0
        tot_hop = 0
        tot_kf = 0

        t2m = t1m = log.timer("Initializing cuguga newton CASSCF", *cput0)
        while (not conv) and imacro < int(getattr(casscf, "max_cycle_macro", 50)):
            imacro += 1
            u, fcivec, norm_gall, stat, dr0 = update_orb_ci(
                casscf,
                mo,
                fcivec,
                eris,
                dr0,
                float(conv_tol_grad) * 0.3,
                verbose=int(getattr(log, "verbose", getattr(casscf, "verbose", 0))),
                weights=weights_use,
                gauge=gauge,
                convention=convention,
                strict_weights=bool(strict_weights),
                enforce_absorb_h1e_direct=bool(enforce_absorb_h1e_direct),
                implementation=implementation,
                ci_update=ci_update,
                ah_mixed_precision=bool(ah_mixed_precision),
            )
            tot_hop += int(stat.tot_hop)
            tot_kf += int(stat.tot_kf)
            t2m = log.timer("update_orb_ci", *t2m)

            eris = None
            mo = casscf.rotate_mo(mo, u, log)
            eris = casscf.ao2mo(mo)
            t2m = log.timer("update eri", *t2m)

            e_tot, e_cas, fcivec = casscf.casci(mo, fcivec, eris, log, locals())
            log.timer("CASCI solver", *t2m)
            t2m = t1m = log.timer(f"macro iter {imacro}", *t1m)

            de, elast = float(e_tot) - elast, float(e_tot)
            if abs(de) < float(tol) and float(norm_gall) < float(conv_tol_grad):
                conv = True

            if dump_chk:
                casscf.dump_chk(locals())

            if callable(callback):
                callback(locals())

        if conv:
            log.info(
                "cuguga newton CASSCF converged in %d macro (%d KF %d Hx) steps", imacro, tot_kf, tot_hop
            )
        else:
            log.info(
                "cuguga newton CASSCF not converged, %d macro (%d KF %d Hx) steps", imacro, tot_kf, tot_hop
            )

        casdm1 = casscf.fcisolver.make_rdm1(fcivec, casscf.ncas, casscf.nelecas)
        if getattr(casscf, "canonicalization", False):
            log.info("CASSCF canonicalization")
            mo, fcivec, mo_energy = casscf.canonicalize(
                mo,
                fcivec,
                eris,
                casscf.sorting_mo_energy,
                casscf.natorb,
                casdm1,
                log,
            )
            if getattr(casscf, "natorb", False):
                ncas = int(casscf.ncas)
                ncore = int(casscf.ncore)
                nocc = ncas + ncore
                occ, _ucas = casscf._eig(-casdm1, ncore, nocc)
                casdm1 = -occ
        else:
            if getattr(casscf, "natorb", False):
                log.warn(
                    "The attribute natorb affects only orbital canonicalization. "
                    "Use mc.cas_natorb_() for natural orbitals in the active space."
                )
            mo_energy = None

        if dump_chk:
            casscf.dump_chk(locals())

        log.timer("cuguga newton CASSCF", *cput0)
        return bool(conv), float(e_tot), e_cas, fcivec, mo, mo_energy


def kernel_newton_inplace(
    casscf: Any,
    mo_coeff: np.ndarray | None = None,
    ci0: Any | None = None,
    callback: Callable[[dict[str, Any]], Any] | None = None,
    *,
    tol: float | None = None,
    conv_tol_grad: float | None = None,
    verbose: int | None = None,
    dump_chk: bool = False,
    weights: Sequence[float] | None = None,
    gauge: str = "none",
    convention: str = "pyscf2",
    strict_weights: bool = False,
    enforce_absorb_h1e_direct: bool = True,
    implementation: str = "internal",
    ci_update: str = "pyscf",
    ah_mixed_precision: bool = False,
) -> tuple[float, Any, Any, np.ndarray, Any]:
    """Run `kernel_newton` and write results back to the `casscf` object.

    Parameters
    ----------
    casscf : Any
        CASSCF object (modified in-place).
    mo_coeff : np.ndarray | None, optional
        Initial MO coefficients. If None, uses casscf.mo_coeff.
    ci0 : Any | None, optional
        Initial CI vector(s). If None, uses casscf.ci.
    callback : Callable | None, optional
        Callback function.
    tol : float | None, optional
        Energy tolerance.
    conv_tol_grad : float | None, optional
        Gradient tolerance.
    verbose : int | None, optional
        Verbosity level.
    dump_chk : bool, optional
        Whether to dump checkpoints.
    weights : Sequence[float] | None, optional
        State-average weights.
    gauge : str, optional
        Gauge regarding CI projection.
    convention : str, optional
        Conventions version.
    strict_weights : bool, optional
        Whether to enforce weight consistency.
    enforce_absorb_h1e_direct : bool, optional
        Whether to force direct H1e absorption.
    implementation : str, optional
        Implementation backend.
    ci_update : str, optional
        CI update method.

    Returns
    -------
    tuple
        (e_tot, e_cas, ci, mo, mo_energy)
    """

    if mo_coeff is None:
        mo_coeff = getattr(casscf, "mo_coeff", None)
    if mo_coeff is None:
        raise ValueError("mo_coeff is required (casscf.mo_coeff is None)")

    tol_use = float(getattr(casscf, "conv_tol", 1e-7)) if tol is None else float(tol)

    conv, e_tot, e_cas, ci, mo, mo_energy = kernel_newton(
        casscf,
        mo_coeff,
        tol=tol_use,
        conv_tol_grad=conv_tol_grad,
        ci0=ci0,
        callback=callback,
        verbose=verbose,
        dump_chk=bool(dump_chk),
        weights=weights,
        gauge=gauge,
        convention=convention,
        strict_weights=bool(strict_weights),
        enforce_absorb_h1e_direct=bool(enforce_absorb_h1e_direct),
        implementation=implementation,
        ci_update=ci_update,
        ah_mixed_precision=bool(ah_mixed_precision),
    )

    casscf.converged = bool(conv)
    casscf.e_tot = float(e_tot)
    casscf.e_cas = e_cas
    casscf.ci = ci
    casscf.mo_coeff = mo
    casscf.mo_energy = mo_energy
    return float(e_tot), e_cas, ci, mo, mo_energy


@dataclass
class OrbitalMicroStats:
    """Micro-iteration stats returned by `rotate_orb_ah`."""

    imic: int = 0
    tot_hop: int = 0
    tot_kf: int = 0
    norm_gorb: float = 0.0


def gen_g_hop_orbital(
    casscf: Any,
    mo: np.ndarray,
    ci0: Any,
    eris: Any,
    *,
    weights: Sequence[float] | None = None,
    strict_weights: bool = False,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray], np.ndarray, Callable[..., tuple[np.ndarray, Callable, np.ndarray]]]:
    """Orbital-only gradient, Hessian-vector product, and diagonal for mc1step-style AH.

    Returns
    -------
    g_orb : np.ndarray
        Packed orbital gradient (scaled by 2).
    h_op_orb : Callable
        Orbital-only Hessian-vector product.
    h_diag_orb : np.ndarray
        Orbital diagonal Hessian (scaled by 2).
    gorb_update : Callable
        Recompute (g_orb, h_op_orb, h_diag_orb) at a new rotation.
    """

    ci0_list = _as_ci_list(ci0)
    nroots = int(len(ci0_list))
    w_info = _resolve_weights(casscf, nroots=nroots, weights=weights, strict=bool(strict_weights))
    w = np.asarray(w_info.weights, dtype=np.float64)

    ncas = int(getattr(casscf, "ncas"))
    ncore = int(getattr(casscf, "ncore"))
    nocc = ncore + ncas
    nmo = int(mo.shape[1])

    fcisolver = getattr(casscf, "fcisolver", None)
    if fcisolver is None:
        raise ValueError("casscf must provide fcisolver")

    # Build per-root RDMs.
    linkstr = _maybe_gen_linkstr(fcisolver, ncas, getattr(casscf, "nelecas"), False)
    try:
        casdm1_r, casdm2_r = fcisolver.states_make_rdm12(ci0_list, ncas, getattr(casscf, "nelecas"), link_index=linkstr)
        casdm1_r = np.asarray(casdm1_r, dtype=np.float64)
        casdm2_r = np.asarray(casdm2_r, dtype=np.float64)
    except AttributeError:
        dm1s_l: list[np.ndarray] = []
        dm2s_l: list[np.ndarray] = []
        for c in ci0_list:
            dm1, dm2 = fcisolver.make_rdm12(c, ncas, getattr(casscf, "nelecas"), link_index=linkstr)
            dm1s_l.append(np.asarray(dm1, dtype=np.float64))
            dm2s_l.append(np.asarray(dm2, dtype=np.float64))
        casdm1_r = np.asarray(dm1s_l, dtype=np.float64)
        casdm2_r = np.asarray(dm2s_l, dtype=np.float64)

    # State-averaged RDMs.
    casdm1 = np.einsum("r,rpq->pq", w, casdm1_r, optimize=True)

    # Build per-root gpq and SA average.
    gpq = _build_gpq_per_root(casscf, mo, ci0_list, eris, strict_weights=bool(strict_weights))
    g_orb_mat = np.einsum("r,rpq->pq", w, gpq, optimize=True)
    g_orb = casscf.pack_uniq_var(g_orb_mat - g_orb_mat.T)
    g_orb = np.asarray(g_orb, dtype=np.float64).ravel() * 2.0

    # Build cache intermediates needed for the orbital Hessian.
    ppaa = getattr(eris, "ppaa", None)
    papa = getattr(eris, "papa", None)
    vhf_c = getattr(eris, "vhf_c", None)
    xp, _on_gpu = _get_xp(ppaa, papa)

    ppaa_arr = _to_xp_f64(ppaa, xp)
    papa_arr = _to_xp_f64(papa, xp)

    casdm1_g = xp.asarray(casdm1, dtype=xp.float64)
    casdm2_g = xp.asarray(np.einsum("r,ruvwx->uvwx", w, casdm2_r, optimize=True), dtype=xp.float64)

    # vhf_a (SA-weighted), vhf_ca, jkcaa, hdm2
    vhf_a = xp.einsum("pquv,uv->pq", ppaa_arr, casdm1_g, optimize=True)
    vhf_a -= 0.5 * xp.einsum("puqv,uv->pq", papa_arr, casdm1_g, optimize=True)
    vhf_c_xp = _to_xp_f64(vhf_c, xp)
    vhf_ca = _to_np_f64(vhf_a) + _to_np_f64(vhf_c_xp)

    # hdm2 (SA-weighted)
    dm2tmp = casdm2_g.transpose(1, 2, 0, 3) + casdm2_g.transpose(0, 2, 1, 3)
    _ppaa_2d = ppaa_arr.reshape(nmo * nmo, ncas * ncas)
    _dm2_2d = casdm2_g.reshape(ncas * ncas, ncas * ncas)
    jtmp_full = (_ppaa_2d @ _dm2_2d).reshape(nmo, nmo, ncas, ncas)
    papa_t = papa_arr.transpose(0, 2, 1, 3)
    _papa_t_2d = papa_t.reshape(nmo * nmo, ncas * ncas)
    _dm2tmp_2d = dm2tmp.reshape(ncas * ncas, ncas * ncas)
    ktmp_full = (_papa_t_2d @ _dm2tmp_2d).reshape(nmo, nmo, ncas, ncas)
    hdm2 = _to_np_f64((jtmp_full + ktmp_full).transpose(0, 2, 1, 3))

    # jkcaa (SA-weighted)
    arange_nocc = xp.arange(nocc)
    ppaa_diag = ppaa_arr[arange_nocc, arange_nocc]
    papa_diag = papa_arr[arange_nocc, :, arange_nocc]
    jkcaa_kernel = 6.0 * papa_diag - 2.0 * ppaa_diag
    jkcaa = _to_np_f64(xp.einsum("pik,ik->pi", jkcaa_kernel, casdm1_g, optimize=True))

    hcore = casscf.get_hcore()
    h1e_mo = _to_np_f64((_to_xp_f64(mo, xp).T @ _to_xp_f64(hcore, xp)) @ _to_xp_f64(mo, xp))
    vhf_c_np = _to_np_f64(vhf_c_xp)

    # dm1_full
    dm1_full = np.zeros((nmo, nmo), dtype=np.float64)
    if ncore:
        idx_c = np.arange(ncore)
        dm1_full[idx_c, idx_c] = 2.0
    dm1_full[ncore:nocc, ncore:nocc] = casdm1

    # paaa
    paaa = ppaa_arr[:, ncore:nocc, :, :]

    # ── Precompute MO-basis 3-index DF integrals for fast block-selective JK ──
    import os as _os_ghop
    _L_t_gpu = None          # (nmo, naux, nmo)
    _L_pq2d_act_gpu = None   # (ncas*nmo, naux)
    _L_pq2d_core_gpu = None  # (ncore*nmo, naux)
    _use_mo_jk = False
    if _os_ghop.environ.get("ASUKA_DISABLE_MO_JK", "0") != "1" and ncore > 0 and _cp is not None:
        _df_B_raw = getattr(casscf, "df_B", None)
        if _df_B_raw is not None:
            try:
                _B_g = _cp.asarray(_df_B_raw, dtype=_cp.float64)
                _C_g = _cp.asarray(mo, dtype=_cp.float64)
                _nao_B = int(_B_g.shape[0])
                _naux_B = int(_B_g.shape[2])
                # Half-transform: H[p,n,Q] = C^T @ B[:,:,Q]
                _H = (_C_g.T @ _B_g.reshape(_nao_B, _nao_B * _naux_B)).reshape(nmo, _nao_B, _naux_B)
                # Full transform: L_t[p,Q,q] = H[p,:,Q] @ C
                _H_t = _cp.ascontiguousarray(_H.transpose(0, 2, 1))  # (nmo, naux, nao)
                _L_t_gpu = (_H_t.reshape(nmo * _naux_B, _nao_B) @ _C_g).reshape(nmo, _naux_B, nmo)
                _L_t_gpu = _cp.ascontiguousarray(_L_t_gpu)
                _L_pq2d_act_gpu = _cp.ascontiguousarray(
                    _L_t_gpu[ncore:nocc].transpose(0, 2, 1).reshape(ncas * nmo, _naux_B)
                )
                _L_pq2d_core_gpu = _cp.ascontiguousarray(
                    _L_t_gpu[:ncore].transpose(0, 2, 1).reshape(ncore * nmo, _naux_B)
                )
                del _H, _H_t, _B_g, _C_g
                _use_mo_jk = True
            except Exception:
                _use_mo_jk = False

    # GPU-resident copies of h_op intermediates (avoids per-call CPU→GPU transfers)
    if _use_mo_jk:
        _h1e_mo_g = _cp.asarray(h1e_mo, dtype=_cp.float64)
        _dm1_full_g = _cp.asarray(dm1_full, dtype=_cp.float64)
        _g_orb_sym_g = _cp.asarray(g_orb_mat + g_orb_mat.T, dtype=_cp.float64)
        _vhf_ca_g = _cp.asarray(vhf_ca, dtype=_cp.float64)
        _vhf_c_g = _cp.asarray(vhf_c_np, dtype=_cp.float64)
        _casdm1_hop_g = _cp.asarray(casdm1, dtype=_cp.float64)
        _hdm2_g = _cp.asarray(hdm2, dtype=_cp.float64)

    # Orbital diagonal Hessian (PySCF Parts 7-6).
    h_diag = np.einsum("ii,jj->ij", h1e_mo, dm1_full) - h1e_mo * dm1_full
    h_diag = h_diag + h_diag.T
    g_diag = np.einsum("r,rpp->p", w, gpq, optimize=True)
    h_diag -= g_diag + g_diag.reshape(-1, 1)
    idx = np.arange(nmo)
    h_diag[idx, idx] += g_diag * 2.0
    v_diag = np.diag(vhf_ca)
    h_diag[:, :ncore] += v_diag.reshape(-1, 1) * 2.0
    h_diag[:ncore] += v_diag * 2.0
    if ncore:
        idxc = np.arange(ncore)
        h_diag[idxc, idxc] -= v_diag[:ncore] * 4.0
    tmp_d = np.einsum("ii,jj->ij", vhf_c_np, casdm1, optimize=True)
    h_diag[:, ncore:nocc] += tmp_d
    h_diag[ncore:nocc, :] += tmp_d.T
    tmp2_d = -vhf_c_np[ncore:nocc, ncore:nocc] * casdm1
    h_diag[ncore:nocc, ncore:nocc] += tmp2_d + tmp2_d.T
    tmp3_d = 6.0 * _to_np_f64(getattr(eris, "k_pc")) - 2.0 * _to_np_f64(getattr(eris, "j_pc"))
    h_diag[ncore:, :ncore] += tmp3_d[ncore:]
    h_diag[:ncore, ncore:] += tmp3_d[ncore:].T
    h_diag[:nocc, ncore:nocc] -= jkcaa
    h_diag[ncore:nocc, :nocc] -= jkcaa.T
    v_diag2 = np.einsum("ijij->ij", hdm2, optimize=True)
    h_diag[ncore:nocc, :] += v_diag2.T
    h_diag[:, ncore:nocc] += v_diag2
    h_diag_orb = casscf.pack_uniq_var(h_diag)
    h_diag_orb = np.asarray(h_diag_orb, dtype=np.float64).ravel() * 2.0

    # Orbital-only Hessian-vector product.
    def _h_op_orb(x_packed: np.ndarray) -> np.ndarray:
        x_packed = np.asarray(x_packed, dtype=np.float64).ravel()
        x1 = casscf.unpack_uniq_var(x_packed)

        if _use_mo_jk:
            # ── Full GPU path: matmuls + block-selective MO-basis JK ──
            _xp = _cp
            x1_g = _xp.asarray(x1, dtype=_xp.float64)

            # Orbital Hessian assembly (GPU)
            x2_g = (_h1e_mo_g @ x1_g) @ _dm1_full_g
            x2_g -= _g_orb_sym_g @ x1_g * 0.5
            if ncore:
                x2_g[:ncore] += (x1_g[:ncore, ncore:] @ _vhf_ca_g[ncore:]) * 2.0
            x2_g[ncore:nocc] += (_casdm1_hop_g @ x1_g[ncore:nocc]) @ _vhf_c_g
            x2_g[:, ncore:nocc] += _xp.einsum(
                "purv,rv->pu", _hdm2_g, x1_g[:, ncore:nocc], optimize=True,
            )

            # Block-selective MO-basis JK with combined K GEMM
            if ncore > 0:
                # dm3_MO: symmetric core↔rest block of x1
                dm3 = _xp.zeros((nmo, nmo), dtype=_xp.float64)
                dm3[:ncore, ncore:] = x1_g[:ncore, ncore:]
                dm3[ncore:, :ncore] = x1_g[:ncore, ncore:].T

                # dm_total = 2*dm3 + dm4
                dm4_h = _xp.zeros((nmo, nmo), dtype=_xp.float64)
                dm4_h[ncore:nocc, :] = _casdm1_hop_g @ x1_g[ncore:nocc, :]
                dm_total = dm3 * 2.0 + dm4_h + dm4_h.T

                _naux = int(_L_t_gpu.shape[1])
                L_t_2d = _L_t_gpu.reshape(nmo, _naux * nmo)
                L_t_act_flat = _L_t_gpu[ncore:nocc].reshape(ncas * _naux, nmo)
                L_t_core_flat = _L_t_gpu[:ncore].reshape(ncore * _naux, nmo)

                # K Step 1: separate small GEMMs for each density
                LDM0 = (L_t_act_flat @ dm3).reshape(ncas, _naux * nmo)
                LDM1 = (L_t_core_flat @ dm_total).reshape(ncore, _naux * nmo)

                # K Step 2: combined GEMM — reads L_t once instead of twice
                K_cat = _xp.vstack([LDM0, LDM1]) @ L_t_2d.T   # (ncas+ncore, nmo)
                K0_act = K_cat[:ncas]                            # (ncas, nmo)
                K1_core = K_cat[ncas:]                           # (ncore, nmo)

                # J0: exploit sparse dm3 (only core↔rest nonzero)
                rho0 = 2.0 * _xp.einsum("iQa,ia->Q", _L_t_gpu[:ncore, :, ncore:], x1_g[:ncore, ncore:], optimize=True)
                J0_act = (_L_pq2d_act_gpu @ rho0).reshape(ncas, nmo)

                # J1: rho1 = 2*rho0 + rho_dm4
                dm4_act = _casdm1_hop_g @ x1_g[ncore:nocc]
                rho_dm4 = 2.0 * (_L_pq2d_act_gpu.T @ dm4_act.ravel())
                rho1 = 2.0 * rho0 + rho_dm4
                J1_core = (_L_pq2d_core_gpu @ rho1).reshape(ncore, nmo)

                x2_g[ncore:nocc] += _casdm1_hop_g @ (J0_act * 2.0 - K0_act)
                x2_g[:ncore, ncore:] += (J1_core * 2.0 - K1_core)[:, ncore:]

            x2 = (x2_g - x2_g.T).get()
        else:
            # ── CPU fallback ──
            x2 = (h1e_mo @ x1) @ dm1_full
            x2 -= (g_orb_mat + g_orb_mat.T) @ x1 * 0.5
            if ncore:
                x2[:ncore] += (x1[:ncore, ncore:] @ vhf_ca[ncore:]) * 2.0
            x2[ncore:nocc] += (casdm1 @ x1[ncore:nocc]) @ vhf_c_np
            x2[:, ncore:nocc] += np.einsum("purv,rv->pu", hdm2, x1[:, ncore:nocc], optimize=True)

            if ncore > 0:
                va, vc = casscf.update_jk_in_ah(mo, x1, casdm1, eris)
                x2[ncore:nocc] += _to_np_f64(va)
                x2[:ncore, ncore:] += _to_np_f64(vc)

            x2 = x2 - x2.T

        return np.asarray(casscf.pack_uniq_var(x2), dtype=np.float64).ravel() * 2.0

    def _gorb_update(u_rot: np.ndarray, ci_new: Any) -> tuple[np.ndarray, Callable, np.ndarray]:
        mo_new = np.asarray(mo, dtype=np.float64) @ np.asarray(u_rot, dtype=np.float64)
        return gen_g_hop_orbital(casscf, mo_new, ci_new, eris, weights=weights, strict_weights=strict_weights)[:3]

    return g_orb, _h_op_orb, h_diag_orb, _gorb_update


def rotate_orb_ah(
    casscf: Any,
    mo: np.ndarray,
    casdm1: np.ndarray,
    casdm2: np.ndarray,
    eris: Any,
    ci0: Any,
    *,
    weights: Sequence[float] | None = None,
    strict_weights: bool = False,
    verbose: int = 0,
) -> tuple[np.ndarray, float, OrbitalMicroStats]:
    """Orbital-only AH micro-iterations for mc1step-style optimization.

    Parameters
    ----------
    casscf : DFNewtonCASSCFAdapter
        CASSCF adapter object.
    mo : np.ndarray
        Current MO coefficients.
    casdm1, casdm2 : np.ndarray
        State-averaged 1- and 2-RDMs from the macro CASCI step.
    eris : DFNewtonERIs
        Integral object.
    ci0 : Any
        Current CI vector(s), needed for gen_g_hop_orbital.
    weights : Sequence[float] | None
        State-average weights.
    strict_weights : bool
        Whether to enforce weight consistency.
    verbose : int
        Verbosity level.

    Returns
    -------
    u : np.ndarray
        Accumulated orbital rotation matrix.
    norm_gorb : float
        Orbital gradient norm at the last keyframe.
    stat : OrbitalMicroStats
        Micro-iteration statistics.
    """

    log = _new_logger(verbose=verbose)
    nmo = int(mo.shape[1])

    max_stepsize = float(getattr(casscf, "max_stepsize", 0.02))
    max_cycle_micro = int(getattr(casscf, "max_cycle_micro", 4))
    kf_interval = int(getattr(casscf, "kf_interval", 4))
    ah_start_tol = float(getattr(casscf, "ah_start_tol", 2.5))
    ah_start_cycle = int(getattr(casscf, "ah_start_cycle", 3))
    ah_conv_tol = float(getattr(casscf, "ah_conv_tol", 1e-12))
    ah_max_cycle = int(getattr(casscf, "ah_max_cycle", 30))
    ah_lindep = float(getattr(casscf, "ah_lindep", 1e-14))
    ah_level_shift = float(getattr(casscf, "ah_level_shift", 1e-8))
    ah_grad_trust_region = float(getattr(casscf, "ah_grad_trust_region", 3.0))
    scale_restoration = float(getattr(casscf, "ah_scale_restoration", 0.5))

    g_orb, h_op_orb, h_diag_orb, gorb_update = gen_g_hop_orbital(
        casscf, mo, ci0, eris, weights=weights, strict_weights=strict_weights,
    )

    norm_gorb = float(np.linalg.norm(g_orb))
    norm_gkf = norm_gorb

    stat = OrbitalMicroStats(imic=0, tot_hop=0, tot_kf=1, norm_gorb=norm_gorb)

    if norm_gorb < 1e-8:
        return np.eye(nmo, dtype=np.float64), norm_gorb, stat

    u = np.eye(nmo, dtype=np.float64)
    dr = np.zeros_like(g_orb)
    x0_guess = g_orb.copy()

    def _make_precond(hdiag: np.ndarray, ls: float) -> Callable[[np.ndarray, float], np.ndarray]:
        def _p(x: np.ndarray, e: float) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64).ravel()
            hd = np.asarray(hdiag, dtype=np.float64).ravel() - (float(e) - float(ls))
            eps = 1e-8
            mask = np.abs(hd) < eps
            if np.any(mask):
                hd = hd.copy()
                hd[mask] = np.copysign(eps, hd[mask])
            out = x / hd
            nrm = float(np.linalg.norm(out))
            if nrm > 0.0:
                out *= 1.0 / nrm
            return out
        return _p

    ikf = 0
    for imic_outer in range(max_cycle_micro):
        precond = _make_precond(h_diag_orb, ah_level_shift)
        g_op = lambda: g_orb  # noqa: E731

        for ah_conv, ihop, w_eig, dxi, hdxi, residual, seig in davidson_cc(
            h_op_orb,
            g_op,
            precond,
            x0_guess,
            tol=ah_conv_tol,
            max_cycle=ah_max_cycle,
            lindep=ah_lindep,
            verbose=log,
        ):
            stat.tot_hop = stat.tot_hop + 1

            norm_residual = float(np.linalg.norm(residual))
            accept_step = (
                bool(ah_conv)
                or int(ihop) == int(ah_max_cycle)
                or ((norm_residual < ah_start_tol) and (int(ihop) >= ah_start_cycle))
                or (float(seig) < ah_lindep)
            )
            if not accept_step:
                continue

            dxi = np.asarray(dxi, dtype=np.float64).ravel()
            hdxi = np.asarray(hdxi, dtype=np.float64).ravel()

            # Clip step to max_stepsize.
            max_abs = float(np.max(np.abs(dxi))) if dxi.size > 0 else 0.0
            if max_abs > max_stepsize:
                scale = max_stepsize / max_abs
                dxi = dxi * scale
                hdxi = hdxi * scale
                log.debug1("Scale orbital AH step by %g", scale)

            # Predict gradient and trust-region check.
            g_trial = np.asarray(g_orb, dtype=np.float64) + np.asarray(hdxi, dtype=np.float64)
            norm_g_trial = float(np.linalg.norm(g_trial))
            if stat.imic >= 2 and norm_g_trial > norm_gkf * ah_grad_trust_region:
                log.debug("Orbital |g| trust fail; stop micro-iterations.")
                break

            # Commit.
            stat.imic += 1
            dr += dxi
            g_orb = g_trial
            norm_gorb = float(np.linalg.norm(g_orb))
            ikf += 1
            x0_guess = dxi.copy()
            log.debug(
                "    orb imic %d  |g|=%3.2e  |dxi|=%3.2e  max=%3.2e",
                stat.imic, norm_gorb, float(np.linalg.norm(dxi)), max_abs,
            )

            if stat.imic >= max_cycle_micro:
                break

            # Keyframe refresh.
            if ikf >= kf_interval or norm_gorb < norm_gkf / ah_grad_trust_region:
                # Apply current rotation and recompute gradient.
                x1_mat = casscf.unpack_uniq_var(dr)
                u_trial = u @ casscf.update_rotate_matrix(
                    casscf.pack_uniq_var(x1_mat)
                )
                g_kf, h_op_kf, h_diag_kf = gorb_update(u_trial, ci0)
                stat.tot_kf += 1
                norm_gkf_new = float(np.linalg.norm(g_kf))

                norm_dg = float(np.linalg.norm(g_kf - g_orb))
                if norm_dg < norm_gorb * ah_grad_trust_region:
                    u = u_trial
                    dr[:] = 0.0
                    g_orb = g_kf
                    h_op_orb = h_op_kf
                    h_diag_orb = h_diag_kf
                    norm_gorb = norm_gkf = norm_gkf_new
                    ikf = 0
                    # Restart Davidson with new operator.
                    break
                else:
                    log.debug("Keyframe trust fail; stop micro-iterations.")
                    break
            break
        else:
            # davidson_cc exhausted without break
            break

        if stat.imic >= max_cycle_micro:
            break

    # Apply final accumulated rotation.
    if float(np.linalg.norm(dr)) > 1e-15:
        u = u @ casscf.update_rotate_matrix(dr)

    stat.norm_gorb = norm_gorb
    return u, norm_gorb, stat


__all__ = [
    "WeightsInfo",
    "NewtonMicroStats",
    "OrbitalMicroStats",
    "compute_ci_gram_inv",
    "extract_rotation",
    "gen_g_hop_internal",
    "gen_g_hop",
    "gen_g_hop_orbital",
    "rotate_orb_ah",
    "kernel_newton",
    "kernel_newton_inplace",
    "update_orb_ci",
    "pack_ci_list",
    "project_ci_root_span",
    "unpack_ci_list",
]
