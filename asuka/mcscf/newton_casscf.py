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
    if xp is np:
        return _to_np_f64(a)
    return xp.asarray(a, dtype=xp.float64)


def _scalar_real_float(a: Any) -> float:
    if hasattr(a, "item"):
        try:
            return float(a.item())
        except Exception:
            pass
    return float(np.asarray(a).reshape(()))


def _norm_f64(a: Any) -> float:
    xp, _ = _get_xp(a)
    arr = xp.asarray(a, dtype=xp.float64).ravel()
    return _scalar_real_float(xp.linalg.norm(arr))


def _env_bool_from_map(env: Any, name: str, default: bool = False) -> bool:
    v = env.get(str(name))
    if v is None:
        return bool(default)
    return str(v).strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}


def _env_int_from_map(env: Any, name: str, default: int) -> int:
    try:
        return int(str(env.get(str(name), str(default))).strip())
    except Exception:
        return int(default)


def _env_float_from_map(env: Any, name: str, default: float) -> float:
    try:
        return float(str(env.get(str(name), str(default))).strip())
    except Exception:
        return float(default)


def _log_cuda_vram_snapshot(cp_mod: Any, label: str, *, sync: bool = False, prefix: str = "[VRAM]") -> None:
    """Best-effort VRAM snapshot logger using CUDA runtime + CuPy pool stats."""

    try:
        if bool(sync):
            cp_mod.cuda.Stream.null.synchronize()
        free_b, total_b = cp_mod.cuda.runtime.memGetInfo()
        pool = cp_mod.get_default_memory_pool()
        used_nvidia = float(total_b - free_b) / 1e9
        pool_used = float(pool.used_bytes()) / 1e9
        pool_total = float(pool.total_bytes()) / 1e9
        print(f"{prefix} {label}: nvidia={used_nvidia:.2f}GB  pool_active={pool_used:.2f}GB  pool_total={pool_total:.2f}GB")
    except Exception:
        return


def _estimate_mojk_precompute_bytes_per_q(
    *,
    mode: str,
    nao: int,
    nmo: int,
    dtype_nbytes: int,
    store_packed: bool,
    use_qp_kernel: bool,
) -> int:
    """Estimate transient bytes per aux index Q for MO-JK precompute."""

    nao_i = int(max(1, nao))
    nmo_i = int(max(1, nmo))
    bpe = int(max(1, dtype_nbytes))
    ntri_mo = int(nmo_i * (nmo_i + 1) // 2)
    mode_s = str(mode).strip().lower()

    if mode_s == "mnq":
        # B_blk + H_blk + H_t_blk + L_blk (+ packed conversion buffers)
        elems_per_q = int(nao_i * nao_i + 2 * nmo_i * nao_i + nmo_i * nmo_i)
        if bool(store_packed):
            elems_per_q += int(nmo_i * nmo_i + ntri_mo)
        return int(max(1, elems_per_q * bpe))

    if mode_s == "qp":
        # X_blk + X_t_blk + L_f_blk + L_blk (+ unpack fallback, packed conversion)
        elems_per_q = int(2 * nao_i * nmo_i + 2 * nmo_i * nmo_i)
        if not bool(use_qp_kernel):
            elems_per_q += int(nao_i * nao_i)
        if bool(store_packed):
            elems_per_q += int(ntri_mo)
        return int(max(1, elems_per_q * bpe))

    return int(max(1, (nao_i * nmo_i + nmo_i * nmo_i) * bpe))


def _choose_mojk_aux_qblk(
    cp_mod: Any,
    env: Any,
    *,
    naux: int,
    default_qblk: int,
    bytes_per_q: int,
    label: str,
    debug: bool = False,
) -> int:
    """Pick aux block size from free VRAM when explicit env override is absent."""

    naux_i = int(max(1, naux))
    explicit_qblk = int(max(0, _env_int_from_map(env, "ASUKA_MO_JK_AUX_BLOCK_NAUX", 0)))
    if explicit_qblk > 0:
        return int(max(1, min(naux_i, explicit_qblk)))

    base_qblk = int(max(1, min(naux_i, default_qblk)))
    if not _env_bool_from_map(env, "ASUKA_MO_JK_AUX_BLOCK_AUTO", default=False):
        return int(base_qblk)

    frac = float(min(0.95, max(0.01, _env_float_from_map(env, "ASUKA_MO_JK_AUX_BLOCK_FREE_FRAC", 0.03))))
    safety_mult = float(max(1.0, _env_float_from_map(env, "ASUKA_MO_JK_AUX_BLOCK_SAFETY_MULT", 8.0)))
    min_qblk = int(max(1, _env_int_from_map(env, "ASUKA_MO_JK_AUX_BLOCK_MIN_NAUX", 8)))
    max_qblk_env = int(max(0, _env_int_from_map(env, "ASUKA_MO_JK_AUX_BLOCK_MAX_NAUX", 0)))

    try:
        free_b, _ = cp_mod.cuda.runtime.memGetInfo()
    except Exception:
        return int(base_qblk)

    budget_b = int(max(1, float(free_b) * frac))
    denom_b = int(max(1, float(max(1, bytes_per_q)) * safety_mult))
    qblk_auto = int(max(1, budget_b // denom_b))

    qblk = int(max(1, min(naux_i, min(base_qblk, max(min_qblk, qblk_auto)))))
    if max_qblk_env > 0:
        qblk = int(max(1, min(qblk, int(max_qblk_env))))

    if bool(debug):
        print(
            f"[VRAM_POLICY] {label}: auto qblk={int(qblk)} "
            f"(base={int(base_qblk)} free={float(free_b)/1e9:.2f}GB "
            f"budget={float(budget_b)/1e9:.2f}GB bytes_per_q={int(bytes_per_q)} "
            f"safety={float(safety_mult):.2f})"
        )
    return int(qblk)


def _choose_mojk_colblk(
    cp_mod: Any,
    env: Any,
    *,
    qblk: int,
    nao: int,
    ncol_total: int,
    default_colblk: int,
    dtype_nbytes: int,
    label: str,
    debug: bool = False,
) -> int:
    """Pick packed-K column block size from free VRAM when override is absent."""

    ncol_i = int(max(1, ncol_total))
    explicit = int(max(0, _env_int_from_map(env, "ASUKA_MO_JK_K_COLBLOCK_PACKED", 0)))
    if explicit > 0:
        return int(max(1, min(ncol_i, explicit)))

    base_colblk = int(max(1, min(ncol_i, default_colblk)))
    if not _env_bool_from_map(env, "ASUKA_MO_JK_K_COLBLOCK_AUTO", default=False):
        return int(base_colblk)

    frac = float(min(0.50, max(0.002, _env_float_from_map(env, "ASUKA_MO_JK_K_COLBLOCK_FREE_FRAC", 0.02))))
    safety_mult = float(max(1.0, _env_float_from_map(env, "ASUKA_MO_JK_K_COLBLOCK_SAFETY_MULT", 2.0)))
    target_mb = float(max(4.0, _env_float_from_map(env, "ASUKA_MO_JK_K_COLBLOCK_TARGET_MB", 48.0)))
    min_colblk = int(max(1, _env_int_from_map(env, "ASUKA_MO_JK_K_COLBLOCK_MIN", 16)))

    try:
        free_b, _ = cp_mod.cuda.runtime.memGetInfo()
    except Exception:
        return int(base_colblk)

    budget_by_frac = float(free_b) * frac
    budget_by_cap = float(target_mb) * (1024.0**2)
    budget_b = int(max(1.0, min(budget_by_frac, budget_by_cap)))
    bytes_per_col = int(max(1, 2 * int(max(1, qblk)) * int(max(1, nao)) * int(max(1, dtype_nbytes))))
    denom_b = int(max(1.0, float(bytes_per_col) * safety_mult))
    col_auto = int(max(1, budget_b // denom_b))
    colblk = int(max(1, min(ncol_i, min(base_colblk, max(min_colblk, col_auto)))))

    if bool(debug):
        print(
            f"[VRAM_POLICY] {label}: auto colblk={int(colblk)} "
            f"(base={int(base_colblk)} free={float(free_b)/1e9:.2f}GB "
            f"budget={float(budget_b)/1e9:.2f}GB bytes_per_col={int(bytes_per_col)} "
            f"safety={float(safety_mult):.2f})"
        )
    return int(colblk)


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


def _safe_eigh(h: Any, s: Any, lindep: float) -> tuple[Any, Any, Any]:
    xp, _ = _get_xp(h, s)
    h_arr = xp.asarray(h, dtype=xp.float64)
    s_arr = xp.asarray(s, dtype=xp.float64)
    seig, t = xp.linalg.eigh(s_arr)
    mask = seig >= float(lindep)
    t = t[:, mask]
    if t.size == 0:
        return xp.zeros((0,), dtype=xp.float64), xp.zeros_like(t), seig
    t = t * (1.0 / xp.sqrt(seig[mask]))
    heff = t.conj().T @ h_arr @ t
    w, v = xp.linalg.eigh(heff)
    return w, t @ v, seig


def _dgemv(v: np.ndarray, m: Sequence[np.ndarray]) -> Any:
    xp, _ = _get_xp(m[0])
    coeff = _to_np_f64(v).ravel()
    out = xp.asarray(m[0], dtype=xp.float64) * float(coeff[0])
    for i, vi in enumerate(coeff[1:]):
        out += float(vi) * xp.asarray(m[i + 1], dtype=xp.float64)
    return out


def _regular_step(
    heff: Any,
    ovlp: Any,
    xs: Sequence[Any],
    ax: Sequence[Any] | None,
    lindep: float,
    log: _SimpleLogger,
    *,
    v_prev: Any | None = None,
    root_v0_min: float = 0.1,
    root_homing: bool = False,
    root_pred_decrease: bool = False,
    root_pred_decrease_tol_rel: float = 1e-3,
    trust_maxabs_orb: float | None = None,
    trust_maxabs_ci: float | None = None,
    ngorb: int | None = None,
    mu_orb: float = 0.0,
    mu_ci: float = 0.0,
    ovlp_orb: Any | None = None,
    ovlp_ci: Any | None = None,
) -> tuple[Any, float, Any, int, Any]:
    xp, _ = _get_xp(heff, ovlp, xs[0])
    w, v, seig = _safe_eigh(heff, ovlp, lindep)
    if w.size == 0 or v.shape[1] == 0:
        return xp.zeros_like(xs[0]), 0.0, xp.zeros((0,), dtype=xp.float64), 0, seig

    nvec = int(v.shape[0])
    nbasis = nvec - 1
    v0_all = _to_np_f64(v[0]).ravel()
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
        vp = _to_np_f64(v_prev).ravel()
        if vp.size < nvec:
            vp = np.pad(vp, (0, nvec - vp.size))
        elif vp.size > nvec:
            vp = vp[:nvec]
        v_prev_use = vp

    # Predicted quadratic decrease in the AH model: g·x + 0.5 x·H x
    # computed in the small subspace using heff/ovlp blocks.
    dE: np.ndarray | None = None
    if root_pred_decrease and nbasis > 0:
        heff_b = _to_np_f64(heff[1:nvec, 1:nvec])
        b = _to_np_f64(heff[1:nvec, 0]).ravel()
        s_full = _to_np_f64(ovlp[1:nvec, 1:nvec])
        dE = np.full((int(v.shape[1]),), np.inf, dtype=np.float64)

        use_block = abs(float(mu_orb) - float(mu_ci)) > 0.0
        s_orb = s_ci = None
        if use_block:
            if ovlp_orb is not None and ovlp_ci is not None:
                s_orb = _to_np_f64(ovlp_orb[1:nvec, 1:nvec])
                s_ci = _to_np_f64(ovlp_ci[1:nvec, 1:nvec])
            else:
                use_block = False  # fallback to scalar-style correction

        for k in cand.tolist():
            v0k = float(v0_all[k])
            if abs(v0k) < 1e-14:
                continue
            ck = _to_np_f64(v[1:nvec, k]).ravel() * (1.0 / v0k)
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
                xk = _dgemv(_to_np_f64(v[1:, k]).ravel() * (1.0 / v0k), xs)
                xk_xp, _ = _get_xp(xk)
                max_orb = _scalar_real_float(xk_xp.max(xk_xp.abs(xk[: int(ngorb)]))) if int(ngorb) > 0 else 0.0
                max_ci = _scalar_real_float(xk_xp.max(xk_xp.abs(xk[int(ngorb) :]))) if int(ngorb) < int(xk.size) else 0.0
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
                    vk = _to_np_f64(v[:, k]).ravel()
                    overlaps.append(abs(float(v_prev_use.conj() @ (_to_np_f64(ovlp) @ vk))))
                sel = int(near[int(np.argmax(overlaps))])
            else:
                sel = int(order[0])
        else:
            sel = int(order[0])
    elif v_prev_use is not None and root_homing:
        # Homing-only selection among candidates.
        overlaps = []
        for k in cand.tolist():
            vk = _to_np_f64(v[:, k]).ravel()
            overlaps.append(abs(float(v_prev_use.conj() @ (_to_np_f64(ovlp) @ vk))))
        sel = int(cand[int(np.argmax(overlaps))])
    else:
        # Legacy: pick the first eigenvector with sufficient v0 component.
        sel = int(cand[0])

    log.debug1("CIAH eigen-sel %d", sel)
    w_t = _scalar_real_float(w[sel])

    v0 = _scalar_real_float(v[0, sel])
    if abs(v0) < 1e-14:
        return xp.zeros_like(xs[0]), w_t, xp.asarray(v[:, sel], dtype=xp.float64), sel, seig
    xtrial = _dgemv(_to_np_f64(v[1:, sel]).ravel() * (1.0 / v0), xs)
    return xp.asarray(xtrial, dtype=xp.float64), w_t, xp.asarray(v[:, sel], dtype=xp.float64), sel, seig


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
    xp, on_gpu = _get_xp(x0, *(tuple(xs) if xs else ()), *(tuple(ax) if ax else ()))
    if dot is np.dot and on_gpu:
        dot = lambda a, b: xp.vdot(a, b)  # noqa: E731

    xs_l = [xp.asarray(v, dtype=xp.float64).ravel() for v in xs]
    ax_l = [xp.asarray(v, dtype=xp.float64).ravel() for v in ax]
    nx = int(len(xs_l))

    x0 = xp.asarray(x0, dtype=xp.float64).ravel()
    problem_size = int(x0.size)
    max_cycle = min(int(max_cycle), problem_size)

    heff = xp.zeros((max_cycle + nx + 1, max_cycle + nx + 1), dtype=xp.float64)
    ovlp = xp.eye(max_cycle + nx + 1, dtype=xp.float64)
    ovlp_orb = None
    ovlp_ci = None
    if root_pred_decrease and (abs(float(mu_orb) - float(mu_ci)) > 0.0) and ngorb is not None:
        # Block overlaps are only needed when mu differs across blocks.
        ovlp_orb = xp.zeros_like(ovlp)
        ovlp_ci = xp.zeros_like(ovlp)
    if nx == 0:
        xs_l.append(x0)
        ax_l.append(xp.asarray(h_op(x0), dtype=xp.float64).ravel())
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
    v_prev: Any | None = None
    best_dx = float("inf")
    n_stagnant = 0
    for istep in range(max_cycle):
        g = xp.asarray(g_op(), dtype=xp.float64).ravel()
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
        s0 = _scalar_real_float(seig[0]) if seig.size else 0.0
        if v_t.size == 0:
            z = xp.zeros_like(x0)
            yield True, istep + 1, w_t, z, z, z, s0
            break
        v_prev = xp.asarray(v_t, dtype=xp.float64).ravel()

        hx = _dgemv(v_t[1:], ax_l)
        dx = hx + g * float(v_t[0]) - w_t * float(v_t[0]) * xtrial
        norm_dx = _norm_f64(dx)
        log.debug1(
            "... AH step %d  index=%d  |dx|=%.5g  eig=%.5g  v[0]=%.5g  lindep=%.5g",
            istep + 1,
            index,
            norm_dx,
            w_t,
            _scalar_real_float(v_t[0]) if v_t.size else 0.0,
            s0,
        )

        v0 = _scalar_real_float(v_t[0]) if v_t.size else 0.0
        if abs(v0) > 1e-14:
            hx = hx * (1.0 / v0)
        else:
            hx = xp.zeros_like(hx)

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
                x_reset = xp.asarray(xtrial, dtype=xp.float64).ravel()
                nrm = _norm_f64(x_reset)
                if nrm > 0.0:
                    x_reset *= 1.0 / nrm
                xs_l = [x_reset]
                ax_l = [xp.asarray(h_op(x_reset), dtype=xp.float64).ravel()]
                heff.fill(0.0)
                ovlp.fill(0.0)
                xp.fill_diagonal(ovlp, 1.0)
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
            x1 = xp.asarray(precond(dx, w_t), dtype=xp.float64).ravel()
            if mgs and xs_l:
                x1_orig = x1.copy()
                for vj in xs_l:
                    x1 = x1 - float(dot(vj.conj(), x1).real) * vj
                nrm = _norm_f64(x1)
                if nrm < float(mgs_eps):
                    x1 = x1_orig
                    nrm = _norm_f64(x1)
                if nrm > 0.0:
                    x1 *= 1.0 / nrm
            xs_l.append(x1)
            ax_l.append(xp.asarray(h_op(x1), dtype=xp.float64).ravel())


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
    ci0_list: list[Any]  # per-root flattened CI vectors
    pack: Callable[[Sequence[Any]], Any]
    unpack: Callable[[Any], list[Any]]


def _pack_ci_getters(ci0: Any) -> _PackedCI:
    ci0_list = _as_ci_list(ci0)
    if len(ci0_list) == 1:
        def _pack(x: Sequence[Any]) -> Any:
            xp, _ = _get_xp(*(x[:1] if len(x) else []))
            return xp.asarray(x[0], dtype=xp.float64).ravel()

        def _unpack(x: Any) -> list[Any]:
            xp, _ = _get_xp(x)
            return [xp.asarray(x, dtype=xp.float64).ravel()]

        return _PackedCI(ci0_list=ci0_list, pack=_pack, unpack=_unpack)

    sizes = [int(getattr(c, "size", np.asarray(_to_np_f64(c)).size)) for c in ci0_list]
    offs: list[int] = [0]
    for s in sizes[:-1]:
        offs.append(offs[-1] + int(s))
    total = int(sum(sizes))

    def _pack(x: Sequence[Any]) -> Any:
        xp, _ = _get_xp(*(x[:1] if len(x) else []))
        parts = [xp.asarray(v, dtype=xp.float64).ravel() for v in x]
        return xp.concatenate(parts, axis=0)

    def _unpack(x: Any) -> list[Any]:
        xp, _ = _get_xp(x)
        x = xp.asarray(x, dtype=xp.float64).ravel()
        if int(x.size) != total:
            raise ValueError("packed CI length mismatch")
        out: list[Any] = []
        off = 0
        for s in sizes:
            out.append(xp.asarray(x[off: off + s], dtype=xp.float64).copy())
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


def _as_ci_list(ci: Any) -> list[Any]:
    if _cp is not None and isinstance(ci, _cp.ndarray):
        c = _cp.asarray(ci, dtype=_cp.float64).ravel()
        if c.size == 0:
            raise ValueError("empty CI vector")
        return [c]
    if isinstance(ci, np.ndarray):
        c = np.asarray(ci, dtype=np.float64).ravel()
        if c.size == 0:
            raise ValueError("empty CI vector")
        return [c]
    if isinstance(ci, (list, tuple)):
        out: list[Any] = []
        for c in ci:
            xp, _ = _get_xp(c)
            arr = xp.asarray(c, dtype=xp.float64).ravel()
            if arr.size == 0:
                raise ValueError("empty CI vector in list")
            out.append(arr)
        if len(out) == 0:
            raise ValueError("empty CI list")
        return out
    raise TypeError(f"unsupported CI type: {type(ci)!r}")


def pack_ci_list(ci_list: Sequence[Any]) -> Any:
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

    if len(ci_list) == 0:
        return np.asarray([], dtype=np.float64)
    xp, _ = _get_xp(*(ci_list[:1]))
    parts = [xp.asarray(c, dtype=xp.float64).ravel() for c in ci_list]
    return xp.concatenate(parts, axis=0)


def unpack_ci_list(x: Any, template_ci_list: Sequence[Any]) -> list[Any]:
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

    xp, _ = _get_xp(x, *(template_ci_list[:1] if len(template_ci_list) else []))
    x = xp.asarray(x, dtype=xp.float64).ravel()
    sizes = [int(getattr(c, "size", np.asarray(_to_np_f64(c)).size)) for c in template_ci_list]
    total = int(sum(sizes))
    if int(x.size) != total:
        raise ValueError(f"packed CI length mismatch: expected {total}, got {int(x.size)}")
    out: list[Any] = []
    off = 0
    for c, sz in zip(template_ci_list, sizes):
        arr = x[off : off + sz].copy()
        shape = getattr(c, "shape", np.asarray(_to_np_f64(c)).shape)
        out.append(arr.reshape(shape))
        off += sz
    return out


def compute_ci_gram_inv(ci_list: Sequence[Any]) -> Any:
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

    xp, _ = _get_xp(*(ci_list[:1] if len(ci_list) else []))
    c_list = [xp.asarray(c, dtype=xp.float64).ravel() for c in ci_list]
    nroots = int(len(c_list))
    if nroots == 0:
        raise ValueError("empty CI list")
    nci = int(c_list[0].size)
    if any(int(c.size) != nci for c in c_list):
        raise ValueError("inconsistent CI sizes across roots")

    cmat = xp.stack(c_list, axis=1)  # (nci, nroots)
    gram = cmat.T @ cmat
    try:
        return xp.linalg.inv(gram)
    except Exception:  # pragma: no cover
        return xp.linalg.pinv(gram)


def project_ci_root_span(
    ci_ref_list: Sequence[Any],
    vec_list: Sequence[Any],
    *,
    gram_inv: Any | None = None,
) -> list[Any]:
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

    xp, _ = _get_xp(*(list(ci_ref_list[:1]) + list(vec_list[:1])))
    ci_ref = [xp.asarray(c, dtype=xp.float64).ravel() for c in ci_ref_list]
    vecs = [xp.asarray(v, dtype=xp.float64) for v in vec_list]
    if len(vecs) != len(ci_ref):
        raise ValueError("vec_list must have the same length as ci_ref_list")
    nroots = int(len(ci_ref))
    nci = int(ci_ref[0].size)
    if any(int(c.size) != nci for c in ci_ref):
        raise ValueError("inconsistent CI sizes across roots")

    if gram_inv is None:
        gram_inv_use = compute_ci_gram_inv(ci_ref)
    else:
        gram_inv_use = xp.asarray(gram_inv, dtype=xp.float64)
        if gram_inv_use.shape != (nroots, nroots):
            raise ValueError("gram_inv has wrong shape")

    cmat = xp.stack(ci_ref, axis=1)  # (nci, nroots)
    out: list[Any] = []
    for v in vecs:
        shape = v.shape
        vflat = v.ravel()
        if int(vflat.size) != nci:
            raise ValueError("CI vector size mismatch in projection")
        coeff = cmat.T @ vflat  # (nroots,)
        vproj = vflat - cmat @ (gram_inv_use @ coeff)
        out.append(xp.ascontiguousarray(vproj.reshape(shape)))
    return out


def _build_ci_active_hamiltonian(casscf: Any, mo: Any, eris: Any) -> CIActiveHamiltonian:
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
    xp, _ = _get_xp(mo, hcore)
    mo_x = _to_xp_f64(mo, xp)
    hcore_x = _to_xp_f64(hcore, xp)
    h1e_mo = _to_np_f64((mo_x.T @ hcore_x) @ mo_x)

    vhf_c = getattr(eris, "vhf_c", None)
    if vhf_c is None:
        raise ValueError("eris must provide attribute 'vhf_c'")
    vhf_c = _to_np_f64(vhf_c)

    h1cas = np.asarray(h1e_mo[ncore:nocc, ncore:nocc], dtype=np.float64) + np.asarray(
        vhf_c[ncore:nocc, ncore:nocc], dtype=np.float64
    )

    ppaa = getattr(eris, "ppaa", None)
    provider = getattr(eris, "eri_provider", None)
    C_act_ref = getattr(eris, "C_act", None)
    if ppaa is not None:
        # PySCF builds eri_cas[a] = ppaa[p=a+ncore][q in active]
        try:
            eri_cas = _to_np_f64(ppaa)[ncore:nocc, ncore:nocc]
        except Exception:
            eri_cas = np.empty((ncas, ncas, ncas, ncas), dtype=np.float64)
            for p in range(ncore, nocc):
                eri_cas[p - ncore] = _to_np_f64(ppaa)[p][ncore:nocc]
    elif provider is not None and C_act_ref is not None:
        eri_cas = _to_np_f64(provider.build_pq_uv(C_act_ref, C_act_ref)).reshape(ncas, ncas, ncas, ncas)
    else:
        raise ValueError("eris must provide either 'ppaa' or provider-backed active ERIs")

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
    h1cas: Any,
    eri_cas: Any,
    ncas: int,
    nelecas: Any,
    ci_list: Sequence[Any],
    link_index: Any | None,
    return_cupy: bool = False,
) -> list[Any]:
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

    xp, _on_gpu = _get_xp(h1cas, eri_cas, *(ci_list[:1] if len(ci_list) else []))
    op = fcisolver.absorb_h1e(h1cas, eri_cas, int(ncas), nelecas, 0.5)
    out: list[Any] = []
    c2e_kw: dict[str, Any] = {}
    if bool(return_cupy) and bool(_on_gpu):
        c2e_kw["return_cupy"] = True
        c2e_kw["contract_2e_backend"] = "cuda"
    for c in ci_list:
        hc = fcisolver.contract_2e(op, c, int(ncas), nelecas, link_index=link_index, **c2e_kw)
        out.append(xp.asarray(hc, dtype=xp.float64).ravel() if bool(return_cupy) and bool(_on_gpu) else np.asarray(hc, dtype=np.float64).ravel())
    return out


def _ci_h_diag(
    fcisolver: Any,
    *,
    h1cas: Any,
    eri_cas: Any,
    ncas: int,
    nelecas: Any,
    return_cupy: bool = False,
) -> Any:
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

    hd = fcisolver.make_hdiag(h1cas, eri_cas, int(ncas), nelecas, return_cupy=bool(return_cupy))
    xp, _on_gpu = _get_xp(h1cas, eri_cas)
    if bool(return_cupy) and bool(_on_gpu):
        return xp.asarray(hd, dtype=xp.float64).ravel()
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
    ci0_list_host = [_to_np_f64(c) for c in ci0_list]
    nroots = int(len(ci0_list))

    # Accept ci1 as list/tuple (per-root) or as a packed vector.
    if isinstance(ci1, (list, tuple)):
        ci1_list = [_to_np_f64(c).ravel() for c in ci1]
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


def compute_mcscf_gradient_vector(
    casscf: Any,
    mo: np.ndarray,
    ci0: Any,
    eris: Any,
    *,
    weights: Sequence[float] | None = None,
    gauge: str = "none",
    strict_weights: bool = False,
    enforce_absorb_h1e_direct: bool = True,
) -> np.ndarray:
    """Build the full Newton-CASSCF gradient vector ``g = [g_orb; g_ci]``.

    This is a gradient-only path (no Hessian/matvec construction). It is useful
    for Z-vector RHS construction where only ``g`` is needed.
    """

    mo_use = mo

    g_orb = _compute_orb_grad_block_from_gpq(
        casscf,
        mo_use,
        ci0,
        eris,
        weights=weights,
        strict_weights=bool(strict_weights),
    )
    g_ci, _ = _compute_ci_grad_and_diag_blocks(
        casscf,
        mo_use,
        ci0,
        eris,
        weights=weights,
        gauge=str(gauge),
        strict_weights=bool(strict_weights),
        enforce_absorb_h1e_direct=bool(enforce_absorb_h1e_direct),
    )
    return np.asarray(np.concatenate([g_orb, g_ci]), dtype=np.float64).ravel()


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
    ci0_list_host = [_to_np_f64(c) for c in ci0_list]
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
    provider = getattr(eris, "eri_provider", None)
    C_act_ref = getattr(eris, "C_act", None)

    # jk response from core density variation (PySCF's in-loop accumulation).
    jk = np.zeros((ncas, ncas), dtype=np.float64)
    if ncore:
        ppaa = getattr(eris, "ppaa", None)
        papa = getattr(eris, "papa", None)
        if ppaa is not None and papa is not None:
            xp, _on_gpu = _get_xp(ppaa, papa)
            ppaa_g = _to_xp_f64(ppaa, xp)
            papa_g = _to_xp_f64(papa, xp)
            ddm_c_g = xp.asarray(ddm_c, dtype=xp.float64)
            jk = _to_np_f64(
                xp.einsum("iq,iquv->uv", ddm_c_g, ppaa_g, optimize=True)
                - 0.5 * xp.einsum("iq,iuqv->uv", ddm_c_g, papa_g, optimize=True)
            )
        elif provider is not None and C_act_ref is not None:
            xp, _ = _get_xp(mo, C_act_ref, vhf_c, getattr(provider, "probe_array", lambda: None)())
            C_act_xp = _to_xp_f64(C_act_ref, xp)
            Jddm, Kddm = provider.jk(xp.asarray(ddm, dtype=xp.float64), want_J=True, want_K=True)
            if Jddm is None or Kddm is None:  # pragma: no cover
                raise RuntimeError("provider.jk returned None while J/K were requested")
            vjk = xp.asarray(Jddm, dtype=xp.float64) - 0.5 * xp.asarray(Kddm, dtype=xp.float64)
            jk = _to_np_f64(C_act_xp.T @ vjk @ C_act_xp)
        else:
            raise ValueError("eris must provide either 'ppaa/papa' or provider-backed J/K for H_co")

    # First-order active-space Hamiltonian pieces induced by orbital rotation x1.
    ppaa = getattr(eris, "ppaa", None)
    if ppaa is not None:
        xp_ci, _ = _get_xp(ppaa)
        paaa_g = _to_xp_f64(ppaa, xp_ci)[:, ncore:nocc]
        ra_g = xp_ci.asarray(ra, dtype=xp_ci.float64)
        aaaa = _to_np_f64(
            (ra_g.T @ paaa_g.reshape(nmo, -1)).reshape((ncas, ncas, ncas, ncas))
        )
    elif provider is not None and C_act_ref is not None:
        xp_ci, _ = _get_xp(mo, C_act_ref, getattr(provider, "probe_array", lambda: None)())
        mo_xp = _to_xp_f64(mo, xp_ci)
        C_act_xp = _to_xp_f64(C_act_ref, xp_ci)
        paaa_g = xp_ci.asarray(provider.build_pu_wx(mo_xp, C_act_xp), dtype=xp_ci.float64).reshape(nmo, ncas, ncas, ncas)
        ra_g = xp_ci.asarray(ra, dtype=xp_ci.float64)
        aaaa = _to_np_f64(
            (ra_g.T @ paaa_g.reshape(nmo, -1)).reshape((ncas, ncas, ncas, ncas))
        )
    else:
        raise ValueError("eris must provide either 'ppaa' or provider-backed mixed ERIs for H_co")
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
    ci0_list: Sequence[Any],
    eris: Any,
    *,
    strict_weights: bool = False,
    return_cupy: bool = False,
) -> Any:
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
    provider = getattr(eris, "eri_provider", None)
    C_act_ref = getattr(eris, "C_act", None)
    ppaa = getattr(eris, "ppaa", None)
    papa = getattr(eris, "papa", None)
    if provider is None and (ppaa is None or papa is None):
        raise ValueError("eris must provide 'ppaa' and 'papa' for gpq builder")
    vhf_c = getattr(eris, "vhf_c", None)
    if vhf_c is None:
        raise ValueError("eris must provide 'vhf_c' for gpq builder")
    xp, _on_gpu = _get_xp(ppaa, papa, mo, C_act_ref, vhf_c)

    link_index = _maybe_gen_linkstr(fcisolver, ncas, nelecas, False)
    if bool(_on_gpu):
        casdm1_list = []
        casdm2_list = []
        for c in ci0_list:
            dm1, dm2 = fcisolver.make_rdm12(
                c,
                ncas,
                nelecas,
                link_index=link_index,
                rdm_backend="cuda",
                return_cupy=True,
                strict_gpu=True,
            )
            casdm1_list.append(xp.asarray(dm1, dtype=xp.float64))
            casdm2_list.append(xp.asarray(dm2, dtype=xp.float64))
        casdm1 = xp.stack(casdm1_list, axis=0)
        casdm2 = xp.stack(casdm2_list, axis=0)
    else:
        ci0_list_host = [_to_np_f64(c) for c in ci0_list]
        try:
            casdm1, casdm2 = fcisolver.states_make_rdm12(ci0_list_host, ncas, nelecas, link_index=link_index)
            casdm1 = np.asarray(casdm1, dtype=np.float64)
            casdm2 = np.asarray(casdm2, dtype=np.float64)
        except AttributeError:
            casdm1_list: list[np.ndarray] = []
            casdm2_list: list[np.ndarray] = []
            for c in ci0_list_host:
                dm1, dm2 = fcisolver.make_rdm12(c, ncas, nelecas, link_index=link_index)
                casdm1_list.append(np.asarray(dm1, dtype=np.float64))
                casdm2_list.append(np.asarray(dm2, dtype=np.float64))
            casdm1 = np.asarray(casdm1_list, dtype=np.float64)
            casdm2 = np.asarray(casdm2_list, dtype=np.float64)

    if casdm1.shape != (nroots, ncas, ncas):
        raise RuntimeError("unexpected casdm1 shape in gpq builder")
    if casdm2.shape != (nroots, ncas, ncas, ncas, ncas):
        raise RuntimeError("unexpected casdm2 shape in gpq builder")

    vhf_c_g = _to_xp_f64(vhf_c, xp)
    casdm1_g = xp.asarray(casdm1, dtype=xp.float64)
    casdm2_g = xp.asarray(casdm2, dtype=xp.float64)

    if provider is not None and C_act_ref is not None:
        mo_xp = _to_xp_f64(mo, xp)
        C_act_xp = _to_xp_f64(C_act_ref, xp)
        vhf_a_list = []
        g_dm2_list = []
        for r in range(nroots):
            D_act = C_act_xp @ casdm1_g[r] @ C_act_xp.T
            Jr, Kr = provider.jk(D_act, want_J=True, want_K=True)
            if Jr is None or Kr is None:  # pragma: no cover
                raise RuntimeError("provider.jk returned None while J/K were requested")
            v_ao = xp.asarray(Jr, dtype=xp.float64) - 0.5 * xp.asarray(Kr, dtype=xp.float64)
            vhf_a_list.append(mo_xp.T @ v_ao @ mo_xp)
            g_dm2_r = provider.contract_pu_wx_dm2(
                C_mo=mo_xp,
                C_act=C_act_xp,
                dm2_wxuv=casdm2_g[r],
                out=None,
                profile=None,
            )
            g_dm2_list.append(xp.asarray(g_dm2_r, dtype=xp.float64))
        vhf_a = xp.stack(vhf_a_list, axis=0)
        g_dm2 = xp.stack(g_dm2_list, axis=0)
    else:
        ppaa_g = _to_xp_f64(ppaa, xp)
        papa_g = _to_xp_f64(papa, xp)
        vhf_a = xp.einsum("pquv,ruv->rpq", ppaa_g, casdm1_g, optimize=True)
        vhf_a -= 0.5 * xp.einsum("puqv,ruv->rpq", papa_g, casdm1_g, optimize=True)
        jtmp_full = xp.einsum("pquv,ruvwx->rpqwx", ppaa_g, casdm2_g, optimize=True)
        g_dm2 = xp.einsum("rpuuv->rpv", jtmp_full[:, :, ncore:nocc, :, :], optimize=True)
    vhf_ca = vhf_a + vhf_c_g[None, :, :]

    hcore = casscf.get_hcore()
    h1e_mo = (_to_xp_f64(mo, xp).T @ _to_xp_f64(hcore, xp)) @ _to_xp_f64(mo, xp)

    gpq = xp.zeros((nroots, nmo, nmo), dtype=xp.float64)
    if ncore:
        gpq[:, :, :ncore] = (h1e_mo[None, :, :ncore] + vhf_ca[:, :, :ncore]) * 2.0

    tmp = h1e_mo[:, ncore:nocc] + vhf_c_g[:, ncore:nocc]
    gpq[:, :, ncore:nocc] = xp.dot(
        xp.asarray(tmp, dtype=xp.float64),
        xp.asarray(casdm1, dtype=xp.float64),
    ).transpose(1, 0, 2)
    gpq[:, :, ncore:nocc] += g_dm2

    if bool(return_cupy) and bool(_on_gpu):
        return xp.asarray(gpq, dtype=xp.float64)
    return _to_np_f64(gpq)


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
    ci0_list_host = [_to_np_f64(c) for c in ci0_list]
    nroots = int(len(ci0_list))
    w_info = _resolve_weights(casscf, nroots=nroots, weights=weights, strict=bool(strict_weights))

    xp, _on_gpu = _get_xp(mo, getattr(eris, "ppaa", None), getattr(eris, "papa", None))
    gpq = _build_gpq_per_root(
        casscf,
        mo,
        ci0_list,
        eris,
        strict_weights=bool(strict_weights),
        return_cupy=bool(_on_gpu),
    )
    g_orb_mat = xp.einsum("r,rpq->pq", xp.asarray(w_info.weights, dtype=xp.float64), xp.asarray(gpq, dtype=xp.float64), optimize=True)
    g_orb_vec = casscf.pack_uniq_var(g_orb_mat - g_orb_mat.T)
    return _to_np_f64(g_orb_vec).ravel() * 2.0


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
    _rdm_kw: dict = dict(link_index=link_index)
    if return_cupy:
        _rdm_kw["return_cupy"] = True
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
    hci0: list[Any]  # per-root flattened
    eci0: np.ndarray  # (nroots,)
    gci0: list[Any]  # per-root residual vectors (unweighted)

    hdiag_all: Any  # packed diag (orb+ci), scaled by 2
    g_all: Any  # packed gradient, scaled by 2
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
    provider = getattr(eris, "eri_provider", None)
    C_act_ref = getattr(eris, "C_act", None)
    ppaa = getattr(eris, "ppaa", None)
    papa = getattr(eris, "papa", None)
    if provider is None and (ppaa is None or papa is None):
        raise ValueError("eris must provide 'ppaa' and 'papa'")
    # Detect array backend from eris — keep contractions on GPU if available.
    xp, _on_gpu = _get_xp(ppaa, papa, mo, C_act_ref, getattr(eris, "vhf_c", None))

    if enforce_absorb_h1e_direct:
        ctx_absorb = _maybe_set_attr(fcisolver, "absorb_h1e_mode", "direct")
    else:
        ctx_absorb = nullcontext(False)

    with ctx_absorb:
        linkstr = _maybe_gen_linkstr(fcisolver, ncas, nelecas, False)
        if bool(_on_gpu):
            dm1s = []
            dm2s = []
            for c in ci0_list:
                dm1, dm2 = fcisolver.make_rdm12(
                    c,
                    ncas,
                    nelecas,
                    link_index=linkstr,
                    rdm_backend="cuda",
                    return_cupy=True,
                    strict_gpu=True,
                )
                dm1s.append(xp.asarray(dm1, dtype=xp.float64))
                dm2s.append(xp.asarray(dm2, dtype=xp.float64))
            casdm1_r = xp.stack(dm1s, axis=0)
            casdm2_r = xp.stack(dm2s, axis=0)
        else:
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

    # Upload RDMs to same device
    casdm1_g = xp.asarray(casdm1_r, dtype=xp.float64)
    casdm2_g = xp.asarray(casdm2_r, dtype=xp.float64)
    dm2tmp_g = casdm2_g.transpose(0, 2, 3, 1, 4) + casdm2_g.transpose(0, 1, 3, 2, 4)

    if provider is not None and C_act_ref is not None:
        mo_xp = _to_xp_f64(mo, xp)
        C_act_xp = _to_xp_f64(C_act_ref, xp)
        C_occ_xp = xp.ascontiguousarray(mo_xp[:, :nocc])
        full_ppaa = xp.asarray(
            provider.build_pq_uv(mo_xp, C_act_xp),
            dtype=xp.float64,
        ).reshape(nmo, nmo, ncas, ncas)
        full_papa = xp.asarray(
            provider.build_pu_qv(mo_xp, C_act_xp),
            dtype=xp.float64,
        ).reshape(nmo, ncas, nmo, ncas)
        paaa = full_ppaa[:, ncore:nocc, :, :]
        eri_cas = full_ppaa[ncore:nocc, ncore:nocc, :, :]
        occ_ppaa = full_ppaa[:nocc, :nocc, :, :]
        occ_papa = full_papa[:nocc, :, :nocc, :]
        del C_occ_xp
        arange_nocc = xp.arange(nocc)
        ppaa_diag = occ_ppaa[arange_nocc, arange_nocc]
        papa_diag = occ_papa[arange_nocc, :, arange_nocc, :]
        jkcaa_kernel = 6.0 * papa_diag - 2.0 * ppaa_diag
        jkcaa_r = xp.einsum("pik,rik->rpi", jkcaa_kernel, casdm1_g, optimize=True)

        vhf_a_list = []
        _ppaa_2d = full_ppaa.reshape(nmo * nmo, ncas * ncas)
        _papa_t_2d = full_papa.transpose(0, 2, 1, 3).reshape(nmo * nmo, ncas * ncas)
        _dm2_2d = casdm2_g.reshape(nroots, ncas * ncas, ncas * ncas)
        _dm2tmp_2d = dm2tmp_g.reshape(nroots, ncas * ncas, ncas * ncas)
        jtmp_full = xp.stack([_ppaa_2d @ _dm2_2d[r] for r in range(nroots)]).reshape(
            nroots, nmo, nmo, ncas, ncas
        )
        ktmp_full = xp.stack([_papa_t_2d @ _dm2tmp_2d[r] for r in range(nroots)]).reshape(
            nroots, nmo, nmo, ncas, ncas
        )
        hdm2_r = (jtmp_full + ktmp_full).transpose(0, 1, 3, 2, 4)
        g_dm2_r = xp.einsum("rpuuv->rpv", jtmp_full[:, :, ncore:nocc, :, :], optimize=True)
        for r in range(nroots):
            D_act_r = C_act_xp @ casdm1_g[r] @ C_act_xp.T
            Jr, Kr = provider.jk(D_act_r, want_J=True, want_K=True)
            if Jr is None or Kr is None:  # pragma: no cover
                raise RuntimeError("provider.jk returned None while J/K were requested")
            v_ao = xp.asarray(Jr, dtype=xp.float64) - 0.5 * xp.asarray(Kr, dtype=xp.float64)
            vhf_a_list.append(mo_xp.T @ v_ao @ mo_xp)
        vhf_a_r = xp.stack(vhf_a_list, axis=0)
    else:
        ppaa_arr = _to_xp_f64(ppaa, xp)  # (nmo,nmo,ncas,ncas)
        papa_arr = _to_xp_f64(papa, xp)  # (nmo,ncas,nmo,ncas)

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

        # hdm2_r and g_dm2_r
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
        hdm2_r = (jtmp_full + ktmp_full).transpose(0, 1, 3, 2, 4)
        g_dm2_r = xp.einsum("rpuuv->rpv", jtmp_full[:, :, ncore:nocc, :, :], optimize=True)

    vhf_c = getattr(eris, "vhf_c", None)
    if vhf_c is None:
        raise ValueError("eris must provide 'vhf_c'")
    vhf_c = _to_xp_f64(vhf_c, xp)
    vhf_ca_r = vhf_c[None, :, :] + vhf_a_r

    hcore = casscf.get_hcore()
    h1e_mo_xp = (_to_xp_f64(mo, xp).T @ _to_xp_f64(hcore, xp)) @ _to_xp_f64(mo, xp)
    vhf_c_np = _to_np_f64(vhf_c)

    gpq_xp = xp.zeros((nroots, nmo, nmo), dtype=xp.float64)
    if ncore:
        gpq_xp[:, :, :ncore] = (h1e_mo_xp[None, :, :ncore] + vhf_ca_r[:, :, :ncore]) * 2.0
    gpq_xp[:, :, ncore:nocc] = xp.dot(
        xp.asarray(h1e_mo_xp[:, ncore:nocc] + vhf_c[:, ncore:nocc], dtype=xp.float64),
        xp.asarray(casdm1_r, dtype=xp.float64),
    ).transpose(1, 0, 2)
    gpq_xp[:, :, ncore:nocc] += g_dm2_r

    w_xp = xp.asarray(w, dtype=xp.float64)
    vhf_ca_xp = xp.einsum("r,rpq->pq", w_xp, vhf_ca_r, optimize=True)
    casdm1_xp = xp.einsum("r,rpq->pq", w_xp, xp.asarray(casdm1_r, dtype=xp.float64), optimize=True)
    jkcaa_xp = xp.einsum("r,rpq->pq", w_xp, jkcaa_r, optimize=True)
    hdm2_xp = xp.einsum("r,rpqst->pqst", w_xp, hdm2_r, optimize=True)

    h1e_mo = h1e_mo_xp if bool(_on_gpu) else _to_np_f64(h1e_mo_xp)
    gpq = gpq_xp if bool(_on_gpu) else _to_np_f64(gpq_xp)
    vhf_ca = vhf_ca_xp if bool(_on_gpu) else _to_np_f64(vhf_ca_xp)
    casdm1 = casdm1_xp if bool(_on_gpu) else _to_np_f64(casdm1_xp)
    jkcaa = jkcaa_xp if bool(_on_gpu) else _to_np_f64(jkcaa_xp)
    hdm2 = hdm2_xp if bool(_on_gpu) else _to_np_f64(hdm2_xp)

    # Active-space Hamiltonian at reference.
    h1cas_0_xp = xp.asarray(h1e_mo_xp[ncore:nocc, ncore:nocc] + vhf_c[ncore:nocc, ncore:nocc], dtype=xp.float64)
    h1cas_0 = h1cas_0_xp if bool(_on_gpu) else _to_np_f64(h1cas_0_xp)
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
            h1cas=h1cas_0 if bool(_on_gpu) else _to_np_f64(h1cas_0),
            eri_cas=eri_cas if bool(_on_gpu) else eri_cas_np,
            ncas=ncas,
            nelecas=nelecas,
            ci_list=ci0_list,
            link_index=linkstrl,
            return_cupy=bool(_on_gpu),
        )
        eci0 = np.asarray(
            [
                _scalar_real_float(
                    xp.dot(
                        _to_xp_f64(c, xp).ravel(),
                        _to_xp_f64(hc, xp).ravel(),
                    )
                )
                for c, hc in zip(ci0_list, hci0)
            ],
            dtype=np.float64,
        )
        gci0 = [
            _to_xp_f64(hc, xp) - _to_xp_f64(c, xp) * float(e)
            if bool(_on_gpu)
            else _to_np_f64(hc) - _to_np_f64(c) * float(e)
            for hc, c, e in zip(hci0, ci0_list, eci0)
        ]

        # Orbital gradient block (via gpq) and CI gradient block.
        g_orb_mat = xp.einsum("r,rpq->pq", w_xp, xp.asarray(gpq, dtype=xp.float64), optimize=True)
        g_orb = casscf.pack_uniq_var(g_orb_mat - g_orb_mat.T)
        ngorb = int(getattr(g_orb, "size", np.asarray(_to_np_f64(g_orb)).size))
        g_ci = ci.pack([g * float(wi) for g, wi in zip(gci0, w)])
        if bool(_on_gpu):
            g_all = xp.concatenate(
                (
                    xp.asarray(g_orb, dtype=xp.float64).ravel() * 2.0,
                    xp.asarray(g_ci, dtype=xp.float64).ravel() * 2.0,
                )
            )
        else:
            g_all = np.hstack(
                (_to_np_f64(g_orb).ravel() * 2.0, _to_np_f64(g_ci).ravel() * 2.0)
            )

        # Orbital diagonal (PySCF parts 7-6).
        orb_xp = xp if bool(_on_gpu) else np
        dm1_full = orb_xp.zeros((nmo, nmo), dtype=orb_xp.float64)
        if ncore:
            idx = orb_xp.arange(ncore)
            dm1_full[idx, idx] = 2.0
        dm1_full[ncore:nocc, ncore:nocc] = orb_xp.asarray(casdm1, dtype=orb_xp.float64)
        h1e_mo_diag = orb_xp.asarray(h1e_mo, dtype=orb_xp.float64)
        vhf_ca_diag = orb_xp.asarray(vhf_ca, dtype=orb_xp.float64)
        vhf_c_diag = orb_xp.asarray(vhf_c_np, dtype=orb_xp.float64)
        casdm1_diag = orb_xp.asarray(casdm1, dtype=orb_xp.float64)
        jkcaa_diag = orb_xp.asarray(jkcaa, dtype=orb_xp.float64)
        hdm2_diag = orb_xp.asarray(hdm2, dtype=orb_xp.float64)
        h_diag = orb_xp.einsum("ii,jj->ij", h1e_mo_diag, dm1_full) - h1e_mo_diag * dm1_full
        h_diag = h_diag + h_diag.T
        g_diag = orb_xp.einsum("r,rpp->p", w_xp, orb_xp.asarray(gpq, dtype=orb_xp.float64), optimize=True)
        h_diag -= g_diag + g_diag.reshape(-1, 1)
        idx = orb_xp.arange(nmo)
        h_diag[idx, idx] += g_diag * 2.0

        v_diag = orb_xp.diag(vhf_ca_diag)
        h_diag[:, :ncore] += v_diag.reshape(-1, 1) * 2.0
        h_diag[:ncore] += v_diag * 2.0
        if ncore:
            idxc = orb_xp.arange(ncore)
            h_diag[idxc, idxc] -= v_diag[:ncore] * 4.0

        tmp = orb_xp.einsum("ii,jj->ij", vhf_c_diag, casdm1_diag, optimize=True)
        h_diag[:, ncore:nocc] += tmp
        h_diag[ncore:nocc, :] += tmp.T
        tmp2 = -vhf_c_diag[ncore:nocc, ncore:nocc] * casdm1_diag
        h_diag[ncore:nocc, ncore:nocc] += tmp2 + tmp2.T

        tmp3 = 6.0 * _to_xp_f64(getattr(eris, "k_pc"), orb_xp) - 2.0 * _to_xp_f64(getattr(eris, "j_pc"), orb_xp)
        h_diag[ncore:, :ncore] += tmp3[ncore:]
        h_diag[:ncore, ncore:] += tmp3[ncore:].T

        h_diag[:nocc, ncore:nocc] -= jkcaa_diag
        h_diag[ncore:nocc, :nocc] -= jkcaa_diag.T

        v_diag2 = orb_xp.einsum("ijij->ij", hdm2_diag, optimize=True)
        h_diag[ncore:nocc, :] += v_diag2.T
        h_diag[:, ncore:nocc] += v_diag2

        h_diag = casscf.pack_uniq_var(h_diag)

        # CI diagonal (PySCF intermediate-normalization fix).
        hd0 = _ci_h_diag(
            fcisolver,
            h1cas=h1cas_0 if bool(_on_gpu) else _to_np_f64(h1cas_0),
            eri_cas=eri_cas if bool(_on_gpu) else eri_cas_np,
            ncas=ncas,
            nelecas=nelecas,
            return_cupy=bool(_on_gpu),
        )
        hci_diag = [
            (_to_xp_f64(hd0, xp) - float(ec) - xp.asarray(gc, dtype=xp.float64) * xp.asarray(c, dtype=xp.float64) * 2.0)
            if bool(_on_gpu)
            else _to_np_f64(hd0) - float(ec) - _to_np_f64(gc) * _to_np_f64(c) * 2.0
            for ec, gc, c in zip(eci0, gci0, ci0_list)
        ]
        hci_diag = [h * float(wi) for h, wi in zip(hci_diag, w)]
        _hci_diag_packed = ci.pack(hci_diag)
        if bool(_on_gpu):
            hdiag_all = xp.concatenate(
                (
                    xp.asarray(h_diag, dtype=xp.float64).ravel() * 2.0,
                    xp.asarray(_hci_diag_packed, dtype=xp.float64).ravel() * 2.0,
                )
            )
        else:
            hdiag_all = np.hstack((_to_np_f64(h_diag).ravel() * 2.0, _to_np_f64(_hci_diag_packed).ravel() * 2.0))

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
        vhf_c=vhf_c if bool(_on_gpu) else vhf_c_np,
        vhf_ca=vhf_ca,
        casdm1=casdm1,
        jkcaa=jkcaa,
        hdm2=hdm2,
        paaa=paaa_np,
        paaa_gpu=paaa_gpu,
        dm1_full=dm1_full,
        h1cas_0=h1cas_0,
        eri_cas=(eri_cas if bool(_on_gpu) else eri_cas_np),
        hci0=hci0,
        eci0=eci0,
        gci0=gci0,
        hdiag_all=(xp.asarray(hdiag_all, dtype=xp.float64).ravel() if bool(_on_gpu) else np.asarray(hdiag_all, dtype=np.float64).ravel()),
        g_all=(xp.asarray(g_all, dtype=xp.float64).ravel() if bool(_on_gpu) else np.asarray(g_all, dtype=np.float64).ravel()),
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
    ci0_list_host = [_to_np_f64(c) for c in ci0_list]
    nroots = int(len(ci0_list))

    # Accept ci1 as list/tuple (per-root) or as packed vector.
    if isinstance(ci1, (list, tuple)):
        ci1_list = [_to_np_f64(c).ravel() for c in ci1]
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

    def g_update(u: Any, fcivec: Any) -> tuple[Any, Callable[[Any], Any], Any]:
        xp_u, _ = _get_xp(mo, u)
        u = _to_xp_f64(u, xp_u)
        if u.ndim != 2 or int(u.shape[0]) != cache.nmo or int(u.shape[1]) != cache.nmo:
            raise ValueError("u must be (nmo,nmo)")
        mo1 = _to_xp_f64(mo, xp_u) @ u
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
        return _to_xp_f64(g1, xp_u).ravel(), hop1, _to_xp_f64(diag1, xp_u)

    with _absorb_ctx():
        linkstrl = _maybe_gen_linkstr(fcisolver, cache.ncas, cache.nelecas, True)
        with _ah_mixed_precision_ctx(fcisolver, ah_mixed_precision):
            op_h0 = fcisolver.absorb_h1e(cache.h1cas_0, cache.eri_cas, cache.ncas, cache.nelecas, 0.5)
    linkstr = _maybe_gen_linkstr(fcisolver, cache.ncas, cache.nelecas, False)

    # ── Closure-scope setup for _h_op_raw ──
    _ppaa_hop = getattr(eris, "ppaa", None)
    _papa_hop = getattr(eris, "papa", None)
    _eri_provider_hop = getattr(eris, "eri_provider", None)
    _C_act_hop = getattr(eris, "C_act", None)
    _provider_probe = None
    if _eri_provider_hop is not None:
        _probe_fn = getattr(_eri_provider_hop, "probe_array", None)
        if callable(_probe_fn):
            _provider_probe = _probe_fn()
    if (_ppaa_hop is None or _papa_hop is None) and (_eri_provider_hop is None or _C_act_hop is None):
        raise ValueError("eris must provide either 'ppaa/papa' or a provider-backed active-space operator")

    # DF factors for memory-efficient per-iteration contractions.
    _L_pu_hop = getattr(eris, "L_pu", None)
    _L_pi_hop = getattr(eris, "L_pi", None)
    _use_df_factors = _L_pu_hop is not None

    # Detect GPU mode from whichever tensors are actually present on the GPU.
    #
    # Important: do NOT unconditionally materialize NumPy copies of large GPU
    # tensors (e.g. ppaa/papa) here; that can trigger massive device->host
    # transfers and destroy Newton-CASSCF performance.
    _hop_xp, _hop_on_gpu = _get_xp(
        _ppaa_hop,
        _papa_hop,
        _L_pu_hop,
        _L_pi_hop,
        getattr(cache, "paaa_gpu", None),
        _provider_probe,
        _C_act_hop,
        mo,
    )
    _supports_return_gpu = hasattr(casscf, "df_B")

    ppaa_dev = papa_dev = None
    if _use_df_factors:
        # Store smaller DF factors; skip ppaa/papa for per-iteration work.
        #
        # Keep CPU copies lazy: only build them if we ever take the CPU path.
        L_pu_cpu = None
        L_pi_cpu = None
        if _hop_on_gpu:
            L_pu_dev = _to_xp_f64(_L_pu_hop, _hop_xp)
            L_pi_dev = _to_xp_f64(_L_pi_hop, _hop_xp) if _L_pi_hop is not None else None
        else:
            L_pu_dev = None
            L_pu_cpu = _to_np_f64(_L_pu_hop)
            if _L_pi_hop is not None:
                L_pi_cpu = _to_np_f64(_L_pi_hop)
            L_pi_dev = None
        # ppaa/papa are NOT captured for the closure.
        ppaa_cpu = None
        papa_cpu = None
    else:
        # Keep CPU copies lazy: only build them if we ever take the CPU path.
        ppaa_cpu = None
        papa_cpu = None

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
        mo_dev = _hop_xp.asarray(mo, dtype=_hop_xp.float64)
        C_act_dev = None if _C_act_hop is None else _hop_xp.asarray(_C_act_hop, dtype=_hop_xp.float64)
        C_core_dev = mo_dev[:, : cache.ncore] if cache.ncore else _hop_xp.zeros((cache.nmo, 0), dtype=_hop_xp.float64)
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
    _mojk_vram_debug = _env_bool_from_map(_os_hop.environ, "ASUKA_MO_JK_VRAM_DEBUG", default=False)
    _mojk_vram_debug_fine = _env_bool_from_map(_os_hop.environ, "ASUKA_MO_JK_VRAM_DEBUG_FINE", default=False)
    _mojk_kblk_logged = {"done": False}
    _mojk_lazy = _env_bool_from_map(_os_hop.environ, "ASUKA_MO_JK_LAZY", default=True)

    # ── Precompute MO-basis 3-index DF integrals L_pq^Q for fast JK in h_op ──
    # L[p,q,Q] = sum_mn C[m,p] * B[m,n,Q] * C[n,q]
    # This replaces AO-basis _df_JK calls (nao^3 * naux) with MO-basis (nmo^3 * naux).
    _L_t_dev = None  # (nmo, naux, nmo) contiguous
    _L_qp_dev = None  # (naux, ntri_mo) packed lower-triangle in MO-pair space
    _use_mo_jk = False
    _mojk_ready = False

    def _ensure_mo_jk_precompute() -> None:
        nonlocal _L_t_dev, _L_qp_dev, _use_mo_jk, _mojk_ready
        if bool(_mojk_ready):
            return
        _mojk_ready = True
        _use_mo_jk = False

        _disable_mo_jk = _os_hop.environ.get("ASUKA_DISABLE_MO_JK", "0") == "1"
        if _disable_mo_jk or (not bool(_hop_on_gpu)) or (not bool(_supports_return_gpu)) or int(cache.ncore) <= 0:
            return
        _df_B_raw = getattr(casscf, "df_B", None)
        if _df_B_raw is None:
            return

        try:
            _xp = _hop_xp
            _B = _xp.asarray(_df_B_raw, dtype=_xp.float64)
            _C = _xp.asarray(mo, dtype=_xp.float64)
            _nmo_C = int(_C.shape[1])
            _nao_C = int(_C.shape[0])
            _allow_packed_mojk = str(_os_hop.environ.get("ASUKA_MO_JK_ALLOW_PACKED_QP", "1")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            _mojk_dtype_s = str(_os_hop.environ.get("ASUKA_MO_JK_DTYPE", "auto")).strip().lower()
            _store_mojk_packed_s = str(_os_hop.environ.get("ASUKA_MO_JK_STORE_PACKED", "auto")).strip().lower()
            if _store_mojk_packed_s in {"1", "true", "yes", "on"}:
                _store_mojk_packed = True
            elif _store_mojk_packed_s in {"0", "false", "no", "off"}:
                _store_mojk_packed = False
            else:
                # Default storage policy:
                # - For packed-Qp AO DF input, prefer packed L_qp storage to keep
                #   VRAM low (dense L_t is ~2x larger in MO-pair space and tends to
                #   dominate gradient VRAM peaks).
                # - For dense AO DF input (mnQ), keep dense L_t.
                if int(getattr(_B, "ndim", 0)) == 2:
                    _store_mojk_packed = True
                else:
                    _store_mojk_packed = False
            _use_qp_apply_kernel = str(_os_hop.environ.get("ASUKA_MO_JK_QP_APPLY_KERNEL", "0")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

            # Dense mnQ path.
            if int(getattr(_B, "ndim", 0)) == 3 and int(_B.shape[0]) == int(_B.shape[1]):
                _nao_B = int(_B.shape[0])
                _naux_B = int(_B.shape[2])
                _use_fp32_mojk = bool(_mojk_dtype_s in {"fp32", "float32", "32"})
                if _mojk_dtype_s in {"", "auto"}:
                    _use_fp32_mojk = False
                _L_dtype = _xp.float32 if _use_fp32_mojk else _xp.float64
                _qblk_default = int(max(1, min(_naux_B, 96)))
                _qblk = _choose_mojk_aux_qblk(
                    _xp,
                    _os_hop.environ,
                    naux=int(_naux_B),
                    default_qblk=int(_qblk_default),
                    bytes_per_q=int(
                        _estimate_mojk_precompute_bytes_per_q(
                            mode="mnq",
                            nao=int(_nao_B),
                            nmo=int(_nmo_C),
                            dtype_nbytes=int(np.dtype(_L_dtype).itemsize),
                            store_packed=bool(_store_mojk_packed),
                            use_qp_kernel=True,
                        )
                    ),
                    label="gen_g_hop_internal MO-JK precompute (mnQ)",
                    debug=bool(_mojk_vram_debug),
                )
                _C_l = _xp.asarray(_C, dtype=_L_dtype)
                if _store_mojk_packed:
                    from asuka.integrals.df_packed_s2 import pack_B_to_Qp  # noqa: PLC0415
                    from asuka.integrals.tri_packed import ntri_from_nao  # noqa: PLC0415

                    _ntri_mo = int(ntri_from_nao(int(_nmo_C)))
                    _L_qp_dev = _xp.empty((_naux_B, _ntri_mo), dtype=_L_dtype)
                else:
                    _L_t_dev = _xp.empty((_nmo_C, _naux_B, _nmo_C), dtype=_L_dtype)
                if bool(_mojk_vram_debug):
                    _log_cuda_vram_snapshot(_xp, "mojk precompute mnQ start", sync=False)
                for _q0 in range(0, _naux_B, _qblk):
                    _q1 = min(_naux_B, _q0 + _qblk)
                    _qb = int(_q1 - _q0)
                    if _qb <= 0:
                        continue
                    if bool(_mojk_vram_debug_fine):
                        _log_cuda_vram_snapshot(_xp, f"mojk precompute mnQ q[{int(_q0)}:{int(_q1)}) start", sync=True)
                    _B_blk = _xp.asarray(_B[:, :, _q0:_q1], dtype=_L_dtype)
                    _H_blk = (_C_l.T @ _B_blk.reshape(_nao_B, _nao_B * _qb)).reshape(_nmo_C, _nao_B, _qb)
                    _H_t_blk = _xp.ascontiguousarray(_H_blk.transpose(0, 2, 1))  # (nmo,qb,nao)
                    _L_blk = (_H_t_blk.reshape(_nmo_C * _qb, _nao_B) @ _C_l).reshape(_nmo_C, _qb, _nmo_C)
                    if _store_mojk_packed:
                        _L_qmn_blk = _xp.ascontiguousarray(_L_blk.transpose(1, 0, 2))  # (qb,nmo,nmo)
                        _L_qp_blk = pack_B_to_Qp(_L_qmn_blk, layout="Qmn", nao=int(_nmo_C))
                        _L_qp_dev[_q0:_q1, :] = _L_qp_blk
                        del _L_qmn_blk, _L_qp_blk
                    else:
                        _L_t_dev[:, _q0:_q1, :] = _L_blk
                    del _B_blk, _H_blk, _H_t_blk, _L_blk
                    if bool(_mojk_vram_debug_fine):
                        _log_cuda_vram_snapshot(_xp, f"mojk precompute mnQ q[{int(_q0)}:{int(_q1)}) done", sync=True)
                if _store_mojk_packed:
                    _L_qp_dev = _xp.ascontiguousarray(_L_qp_dev)
                else:
                    _L_t_dev = _xp.ascontiguousarray(_L_t_dev)
                if bool(_mojk_vram_debug):
                    _log_cuda_vram_snapshot(_xp, "mojk precompute mnQ done", sync=True)
                del _C_l
                _use_mo_jk = True

            # Packed Qp path.
            elif int(getattr(_B, "ndim", 0)) == 2 and bool(_allow_packed_mojk):
                from asuka.integrals.df_packed_s2 import apply_Qp_to_C_block, unpack_Qp_to_Qmn_block  # noqa: PLC0415
                from asuka.integrals.tri_packed import ntri_from_nao  # noqa: PLC0415

                _naux_B, _ntri_B = map(int, _B.shape)
                _nao_B = int(_nao_C)
                if int(_ntri_B) == int(ntri_from_nao(int(_nao_B))):
                    _use_fp32_mojk = bool(_mojk_dtype_s in {"fp32", "float32", "32"})
                    if _mojk_dtype_s in {"", "auto"}:
                        _use_fp32_mojk = False
                    _L_dtype = _xp.float32 if _use_fp32_mojk else _xp.float64
                    _qblk_default = int(max(1, min(_naux_B, 48)))
                    _qblk = _choose_mojk_aux_qblk(
                        _xp,
                        _os_hop.environ,
                        naux=int(_naux_B),
                        default_qblk=int(_qblk_default),
                        bytes_per_q=int(
                            _estimate_mojk_precompute_bytes_per_q(
                                mode="qp",
                                nao=int(_nao_B),
                                nmo=int(_nmo_C),
                                dtype_nbytes=int(np.dtype(_L_dtype).itemsize),
                                store_packed=bool(_store_mojk_packed),
                                use_qp_kernel=bool(_use_qp_apply_kernel and _L_dtype == _xp.float64),
                            )
                        ),
                        label="gen_g_hop_internal MO-JK precompute (Qp)",
                        debug=bool(_mojk_vram_debug),
                    )
                    _C_l = _xp.asarray(_C, dtype=_L_dtype)
                    if _store_mojk_packed:
                        from asuka.integrals.df_packed_s2 import pack_B_to_Qp  # noqa: PLC0415
                        from asuka.integrals.tri_packed import ntri_from_nao  # noqa: PLC0415

                        _ntri_mo = int(ntri_from_nao(int(_nmo_C)))
                        _L_qp_dev = _xp.empty((_naux_B, _ntri_mo), dtype=_L_dtype)
                    else:
                        _L_t_dev = _xp.empty((_nmo_C, _naux_B, _nmo_C), dtype=_L_dtype)
                    if bool(_mojk_vram_debug):
                        _log_cuda_vram_snapshot(_xp, "mojk precompute Qp start", sync=False)
                    for _q0 in range(0, _naux_B, _qblk):
                        _q1 = min(_naux_B, _q0 + _qblk)
                        _qb = int(_q1 - _q0)
                        if _qb <= 0:
                            continue
                        if bool(_mojk_vram_debug_fine):
                            _log_cuda_vram_snapshot(_xp, f"mojk precompute Qp q[{int(_q0)}:{int(_q1)}) start", sync=True)
                        if _use_qp_apply_kernel and _L_dtype == _xp.float64:
                            _X_blk = apply_Qp_to_C_block(
                                _B,
                                _C_l,
                                nao=int(_nao_B),
                                q0=int(_q0),
                                q_count=int(_qb),
                            )  # (qb,nao,nmo)
                        else:
                            _B_qmn = unpack_Qp_to_Qmn_block(_B, nao=int(_nao_B), q0=int(_q0), q_count=int(_qb))
                            _B_qmn = _xp.asarray(_B_qmn, dtype=_L_dtype)
                            # Avoid CuPy's batched-matmul workspace spikes by reshaping
                            # the (qb,nao,nao) block into a single 2D GEMM.
                            _X_f = _B_qmn.reshape(_qb * _nao_B, _nao_B) @ _C_l  # (qb*nao,nmo)
                            _X_blk = _X_f.reshape(_qb, _nao_B, _nmo_C)  # (qb,nao,nmo)
                            del _X_f
                            del _B_qmn
                        # Build all q in one GEMM to avoid batched-matmul workspace spikes:
                        #   X_t: (nao, qb, nmo) -> (nao, qb*nmo)
                        #   L_f: (nmo, nao) @ (nao, qb*nmo) = (nmo, qb*nmo)
                        #   L_qmn: (qb, nmo, nmo)
                        _X_t_blk = _xp.ascontiguousarray(_X_blk.transpose(1, 0, 2))
                        _L_f_blk = _C_l.T @ _X_t_blk.reshape(_nao_B, _qb * _nmo_C)
                        if _store_mojk_packed and _L_dtype == _xp.float64:
                            # Fast path: pack directly from the L_f block without
                            # materializing (qb,nmo,nmo) or an extra pack output.
                            from asuka.integrals.df_packed_s2 import pack_Lf_block_to_Qp  # noqa: PLC0415

                            pack_Lf_block_to_Qp(
                                _L_f_blk,
                                _L_qp_dev,
                                naux=int(_naux_B),
                                nao=int(_nmo_C),
                                q0=int(_q0),
                                q_count=int(_qb),
                                threads=256,
                                sync=False,
                            )
                            del _X_blk, _X_t_blk, _L_f_blk
                        else:
                            _L_blk = _xp.ascontiguousarray(_L_f_blk.reshape(_nmo_C, _qb, _nmo_C).transpose(1, 0, 2))
                            del _X_t_blk, _L_f_blk
                            if _store_mojk_packed:
                                _L_qp_blk = pack_B_to_Qp(_L_blk, layout="Qmn", nao=int(_nmo_C))
                                _L_qp_dev[_q0:_q1, :] = _L_qp_blk
                                del _L_qp_blk
                            else:
                                _L_t_dev[:, _q0:_q1, :] = _L_blk.transpose(1, 0, 2)
                            del _X_blk, _L_blk
                        if bool(_mojk_vram_debug_fine):
                            _log_cuda_vram_snapshot(_xp, f"mojk precompute Qp q[{int(_q0)}:{int(_q1)}) done", sync=True)
                    if _store_mojk_packed:
                        _L_qp_dev = _xp.ascontiguousarray(_L_qp_dev)
                    else:
                        _L_t_dev = _xp.ascontiguousarray(_L_t_dev)
                    if bool(_mojk_vram_debug):
                        _log_cuda_vram_snapshot(_xp, "mojk precompute Qp done", sync=True)
                    del _C_l
                    _use_mo_jk = True
            del _B
        except Exception:
            # Best-effort: fall back to AO-basis JK without failing the Newton/Z solve.
            _L_t_dev = None
            _L_qp_dev = None
            _use_mo_jk = False

    def _release_mo_jk_precompute() -> None:
        nonlocal _L_t_dev, _L_qp_dev, _use_mo_jk, _mojk_ready
        # Also drop large DF/ERI tensors captured by the Hessian matvec closure.
        #
        # This hook is used by the DF gradient driver after the Z-vector solve,
        # and the Hessian operator is stripped immediately afterward. Keeping
        # these buffers alive would otherwise inflate the DF contraction peak
        # (notably `L_pi`).
        nonlocal _ppaa_hop, _papa_hop, _L_pu_hop, _L_pi_hop
        nonlocal ppaa_cpu, papa_cpu
        nonlocal L_pu_cpu, L_pi_cpu
        nonlocal ppaa_dev, papa_dev
        nonlocal L_pu_dev, L_pi_dev
        nonlocal ci0_list_dev, hci0_dev, eci0_dev
        nonlocal h1e_mo_dev, vhf_c_dev, vhf_ca_dev
        nonlocal casdm1_dev, hdm2_dev, dm1_full_dev, gpq_dev, weights_dev
        nonlocal paaa_dev
        nonlocal h1cas_0_dev, eri_cas_dev, op_h0_dev
        _L_t_dev = None
        _L_qp_dev = None
        _use_mo_jk = False
        _mojk_ready = False

        _ppaa_hop = None
        _papa_hop = None
        _L_pu_hop = None
        _L_pi_hop = None
        ppaa_cpu = None
        papa_cpu = None
        L_pu_cpu = None
        L_pi_cpu = None
        ppaa_dev = None
        papa_dev = None
        L_pu_dev = None
        L_pi_dev = None
        ci0_list_dev = None
        hci0_dev = None
        eci0_dev = None
        h1e_mo_dev = None
        vhf_c_dev = None
        vhf_ca_dev = None
        casdm1_dev = None
        hdm2_dev = None
        dm1_full_dev = None
        gpq_dev = None
        weights_dev = None
        paaa_dev = None
        h1cas_0_dev = None
        eri_cas_dev = None
        op_h0_dev = None

    if not bool(_mojk_lazy):
        _ensure_mo_jk_precompute()

    def _h_op_raw(x):
        # Use GPU path when GPU tensors exist in closure, regardless of
        # input vector type.  davidson_cc passes NumPy vectors, but the
        # expensive operations (sigma vectors, trans-RDMs, einsums over
        # ppaa/papa) benefit hugely from running on GPU.  The parameter
        # vector upload (~25K doubles) is negligible.
        _input_is_gpu = bool(_hop_on_gpu) and isinstance(x, _hop_xp.ndarray)
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
            _mo_ref = mo_dev
            _C_act_ref = C_act_dev
            _C_core_ref = C_core_dev
        else:
            paaa = cache.paaa
            if _use_df_factors:
                # Lazily materialize CPU copies only when needed.
                nonlocal L_pu_cpu, L_pi_cpu
                if L_pu_cpu is None:
                    L_pu_cpu = _to_np_f64(_L_pu_hop)
                if _L_pi_hop is not None and L_pi_cpu is None:
                    L_pi_cpu = _to_np_f64(_L_pi_hop)
                _L_pu, _L_pi = L_pu_cpu, L_pi_cpu
            elif _eri_provider_hop is not None and _C_act_hop is not None:
                _L_pu = _L_pi = None
            else:
                nonlocal ppaa_cpu, papa_cpu
                if ppaa_cpu is None:
                    ppaa_cpu = _to_np_f64(_ppaa_hop)
                if papa_cpu is None:
                    papa_cpu = _to_np_f64(_papa_hop)
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
            _mo_ref = mo
            _C_act_ref = _C_act_hop
            _C_core_ref = mo[:, : cache.ncore] if cache.ncore else np.zeros((cache.nmo, 0), dtype=np.float64)

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
        _c2e_kw: dict = dict(link_index=linkstrl)
        if on_gpu:
            _c2e_kw["return_cupy"] = True
            _c2e_kw["contract_2e_backend"] = "cuda"
        with _ah_mixed_precision_ctx(fcisolver, ah_mixed_precision), _absorb_ctx():
            hci1 = [
                (
                    _to_xp_f64(
                        fcisolver.contract_2e(_op_h0, c1, cache.ncas, cache.nelecas, **_c2e_kw),
                        xp,
                    ).ravel()
                    if on_gpu
                    else _to_np_f64(
                        fcisolver.contract_2e(_op_h0, c1, cache.ncas, cache.nelecas, **_c2e_kw)
                    ).ravel()
                )
                for c1 in ci1_list
            ]
        # Intermediate-normalisation correction (zero float() calls on GPU).
        hci1 = [
            _to_xp_f64(hc1, xp) - _to_xp_f64(c1, xp) * ec0
            if on_gpu
            else _to_np_f64(hc1) - _to_np_f64(c1) * ec0
            for hc1, c1, ec0 in zip(hci1, ci1_list, _eci0)
        ]
        hci1 = [
            _to_xp_f64(hc1, xp)
            - (_to_xp_f64(hc0, xp) - _to_xp_f64(c0, xp) * ec0) * xp.dot(_to_xp_f64(c0, xp), _to_xp_f64(c1, xp))
            for hc1, hc0, c0, ec0, c1 in zip(hci1, _hci0, _ci0, _eci0, ci1_list)
        ]
        hci1 = [
            _to_xp_f64(hc1, xp)
            - _to_xp_f64(c0, xp) * xp.dot(_to_xp_f64(hc0, xp) - _to_xp_f64(c0, xp) * ec0, _to_xp_f64(c1, xp))
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
            elif _eri_provider_hop is not None and _C_act_ref is not None:
                D_act = _C_act_ref @ tdm1 @ _C_act_ref.T
                Ja, Ka = _eri_provider_hop.jk(D_act, want_J=True, want_K=True)
                if Ja is None or Ka is None:  # pragma: no cover
                    raise RuntimeError("provider.jk returned None while J/K were requested")
                v_ao = xp.asarray(Ja, dtype=xp.float64) - 0.5 * xp.asarray(Ka, dtype=xp.float64)
                vhf_a = (_mo_ref.T @ v_ao) @ _C_core_ref
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
        elif _eri_provider_hop is not None and _C_act_ref is not None:
            Jc, Kc = _eri_provider_hop.jk(ddm_c, want_J=True, want_K=True)
            if Jc is None or Kc is None:  # pragma: no cover
                raise RuntimeError("provider.jk returned None while J/K were requested")
            vjk = xp.asarray(Jc, dtype=xp.float64) - 0.5 * xp.asarray(Kc, dtype=xp.float64)
            jk = _C_act_ref.T @ vjk @ _C_act_ref
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
        if on_gpu and cache.ncore > 0:
            _ensure_mo_jk_precompute()
        if _HOP_PROFILE:
            if on_gpu: xp.cuda.Stream.null.synchronize()
            _t4_hop = time.perf_counter()
        if _use_mo_jk and on_gpu and cache.ncore > 0:
            # MO-basis JK using precomputed L_pq^Q — avoids nao^3 contractions.
            _nmo_L = cache.nmo
            _ncore_L = cache.ncore
            _nocc_L = cache.nocc
            _ncas_L = _nocc_L - _ncore_L

            # dm3_MO: symmetric core-rest block of x1.
            _dm3 = xp.zeros((_nmo_L, _nmo_L), dtype=xp.float64)
            _dm3[:_ncore_L, _ncore_L:] = x1[:_ncore_L, _ncore_L:]
            _dm3[_ncore_L:, :_ncore_L] = x1[:_ncore_L, _ncore_L:].T
            # dm4_MO: active-all block weighted by casdm1 (symmetric completion).
            _dm4_h = xp.zeros((_nmo_L, _nmo_L), dtype=xp.float64)
            _dm4_h[_ncore_L:_nocc_L, :] = _casdm1 @ x1[_ncore_L:_nocc_L, :]
            _dm_total = _dm3 * 2.0 + _dm4_h + _dm4_h.T

            if _L_qp_dev is not None:
                # Memory-saving path: keep MO DF tensor packed in MO-pair space.
                from asuka.hf import df_scf as _df_scf  # noqa: PLC0415
                from asuka.integrals.df_packed_s2 import extract_Qp_rows_cols_block  # noqa: PLC0415

                _L_dtype = _L_qp_dev.dtype
                _use_fp32_l = bool(_L_dtype != xp.float64)
                if _use_fp32_l:
                    _dm3_jk = _dm3.astype(_L_dtype, copy=False)
                    _dm_total_jk = _dm_total.astype(_L_dtype, copy=False)
                else:
                    _dm3_jk = _dm3
                    _dm_total_jk = _dm_total

                _naux_L = int(_L_qp_dev.shape[0])
                try:
                    _k_qblk = int(str(_os_hop.environ.get("ASUKA_MO_JK_K_QBLOCK_PACKED", "")).strip() or "0")
                except Exception:
                    _k_qblk = 0
                if _k_qblk <= 0:
                    try:
                        _k_qblk = int(str(_os_hop.environ.get("ASUKA_DF_JK_K_QBLOCK_PACKED", "")).strip() or "0")
                    except Exception:
                        _k_qblk = 0
                if _k_qblk <= 0:
                    _k_qblk = 128
                _k_qblk = int(max(1, min(_naux_L, _k_qblk)))

                _k_cblk = _choose_mojk_colblk(
                    xp,
                    _os_hop.environ,
                    qblk=int(_k_qblk),
                    nao=int(_nmo_L),
                    ncol_total=int(_nmo_L),
                    default_colblk=256,
                    dtype_nbytes=int(np.dtype(_L_dtype).itemsize),
                    label="gen_g_hop_internal packed-K",
                    debug=bool(_mojk_vram_debug and not bool(_mojk_kblk_logged["done"])),
                )
                _mojk_kblk_logged["done"] = True

                # J blocks (avoid full J build; exploit dm3/dm4 structure like the dense-L_t path).
                _nrest_L = int(_nmo_L - _ncore_L)
                _x_core_rest = x1[:_ncore_L, _ncore_L:]  # (ncore, nrest)
                _dm4_act = _casdm1 @ x1[_ncore_L:_nocc_L, :]  # (ncas, nmo)
                _J0_act = xp.zeros((_ncas_L, _nmo_L), dtype=xp.float64)
                _J1_core_rest = xp.zeros((_ncore_L, _nrest_L), dtype=xp.float64)
                for _q0 in range(0, _naux_L, _k_qblk):
                    _q1 = min(_naux_L, _q0 + _k_qblk)
                    _qb = int(_q1 - _q0)
                    if _qb <= 0:
                        continue

                    _L_rest_core = extract_Qp_rows_cols_block(
                        _L_qp_dev,
                        nao=int(_nmo_L),
                        q0=int(_q0),
                        q_count=int(_qb),
                        row0=int(_ncore_L),
                        row_count=int(_nrest_L),
                        col0=0,
                        col_count=int(_ncore_L),
                    )
                    _rho0 = 2.0 * xp.einsum("qai,ia->q", _L_rest_core, _x_core_rest, optimize=True)
                    del _L_rest_core

                    _L_act_all = extract_Qp_rows_cols_block(
                        _L_qp_dev,
                        nao=int(_nmo_L),
                        q0=int(_q0),
                        q_count=int(_qb),
                        row0=int(_ncore_L),
                        row_count=int(_ncas_L),
                        col0=0,
                        col_count=int(_nmo_L),
                    )
                    _J0_act += xp.einsum("q,qpm->pm", _rho0, _L_act_all, optimize=True)
                    _rho_dm4 = 2.0 * xp.einsum("qpm,pm->q", _L_act_all, _dm4_act, optimize=True)
                    del _L_act_all

                    _rho1 = _rho0 * 2.0 + _rho_dm4
                    del _rho0, _rho_dm4

                    _L_core_rest = extract_Qp_rows_cols_block(
                        _L_qp_dev,
                        nao=int(_nmo_L),
                        q0=int(_q0),
                        q_count=int(_qb),
                        row0=0,
                        row_count=int(_ncore_L),
                        col0=int(_ncore_L),
                        col_count=int(_nrest_L),
                    )
                    _J1_core_rest += xp.einsum("q,qim->im", _rho1, _L_core_rest, optimize=True)
                    del _L_core_rest, _rho1

                _K0_act = _df_scf._df_K_qblocked_Qp_rows_cols(  # noqa: SLF001
                    _L_qp_dev,
                    _dm3_jk,
                    nao=int(_nmo_L),
                    row0=int(_ncore_L),
                    row_count=int(_ncas_L),
                    col0=0,
                    col_count=int(_nmo_L),
                    q_block=int(_k_qblk),
                    col_block=int(_k_cblk),
                )
                _K1_core_rest = _df_scf._df_K_qblocked_Qp_rows_cols(  # noqa: SLF001
                    _L_qp_dev,
                    _dm_total_jk,
                    nao=int(_nmo_L),
                    row0=0,
                    row_count=int(_ncore_L),
                    col0=int(_ncore_L),
                    col_count=int(_nmo_L - _ncore_L),
                    q_block=int(_k_qblk),
                    col_block=int(_k_cblk),
                )
                if _use_fp32_l:
                    _K0_act = _K0_act.astype(xp.float64, copy=False)
                    _K1_core_rest = _K1_core_rest.astype(xp.float64, copy=False)

                va = _casdm1 @ (_J0_act * 2.0 - _K0_act)
                vc = _J1_core_rest * 2.0 - _K1_core_rest
            else:
                # Full L_t path: fastest but uses more memory.
                _naux_L = int(_L_t_dev.shape[1])
                _L_dtype = _L_t_dev.dtype
                _use_fp32_l = bool(_L_dtype != xp.float64)
                _L_t_2d = _L_t_dev.reshape(_nmo_L, _naux_L * _nmo_L)  # (nmo, naux*nmo)
                _L_t_act_flat = _L_t_dev[_ncore_L:_nocc_L].reshape(_ncas_L * _naux_L, _nmo_L)
                _L_t_core_flat = _L_t_dev[:_ncore_L].reshape(_ncore_L * _naux_L, _nmo_L)

                if _use_fp32_l:
                    _dm3_jk = _dm3.astype(_L_dtype, copy=False)
                    _dm_total_jk = _dm_total.astype(_L_dtype, copy=False)
                    _x_core_rest = x1[:_ncore_L, _ncore_L:].astype(_L_dtype, copy=False)
                    _x_act = x1[_ncore_L:_nocc_L].astype(_L_dtype, copy=False)
                    _casdm1_jk = _casdm1.astype(_L_dtype, copy=False)
                else:
                    _dm3_jk = _dm3
                    _dm_total_jk = _dm_total
                    _x_core_rest = x1[:_ncore_L, _ncore_L:]
                    _x_act = x1[_ncore_L:_nocc_L]
                    _casdm1_jk = _casdm1

                _LDM0 = (_L_t_act_flat @ _dm3_jk).reshape(_ncas_L, _naux_L * _nmo_L)
                _LDM1 = (_L_t_core_flat @ _dm_total_jk).reshape(_ncore_L, _naux_L * _nmo_L)
                try:
                    _k_accum_qblk = int(str(_os_hop.environ.get("ASUKA_MO_JK_K_ACCUM_QBLOCK", "")).strip() or "0")
                except Exception:
                    _k_accum_qblk = 0
                if _k_accum_qblk <= 0:
                    _k_accum_qblk = int(max(1, min(_naux_L, 256)))
                else:
                    _k_accum_qblk = int(max(1, min(_naux_L, _k_accum_qblk)))

                _K0_act = xp.zeros((_ncas_L, _nmo_L), dtype=_LDM0.dtype)
                _K1_core = xp.zeros((_ncore_L, _nmo_L), dtype=_LDM1.dtype)
                for _q0 in range(0, _naux_L, _k_accum_qblk):
                    _q1 = min(_naux_L, _q0 + _k_accum_qblk)
                    _qb = int(_q1 - _q0)
                    if _qb <= 0:
                        continue
                    _off0 = int(_q0 * _nmo_L)
                    _off1 = int(_q1 * _nmo_L)
                    _L_blk_t = _L_t_dev[:, _q0:_q1, :].reshape(_nmo_L, _qb * _nmo_L).T
                    _K0_act += _LDM0[:, _off0:_off1] @ _L_blk_t
                    _K1_core += _LDM1[:, _off0:_off1] @ _L_blk_t

                _rho0 = 2.0 * xp.einsum(
                    "iQa,ia->Q",
                    _L_t_dev[:_ncore_L, :, _ncore_L:],
                    _x_core_rest,
                    optimize=True,
                )
                _J0_act = xp.einsum("pQq,Q->pq", _L_t_dev[_ncore_L:_nocc_L], _rho0, optimize=True)
                _dm4_act = _casdm1_jk @ _x_act
                _rho_dm4 = 2.0 * xp.einsum("pQq,pq->Q", _L_t_dev[_ncore_L:_nocc_L], _dm4_act, optimize=True)
                _rho1 = 2.0 * _rho0 + _rho_dm4
                _J1_core = xp.einsum("pQq,Q->pq", _L_t_dev[:_ncore_L], _rho1, optimize=True)

                if _use_fp32_l:
                    _J0_act = _J0_act.astype(xp.float64, copy=False)
                    _K0_act = _K0_act.astype(xp.float64, copy=False)
                    _J1_core = _J1_core.astype(xp.float64, copy=False)
                    _K1_core = _K1_core.astype(xp.float64, copy=False)

                va = _casdm1 @ (_J0_act * 2.0 - _K0_act)
                vc = (_J1_core * 2.0 - _K1_core)[:, _ncore_L:]
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
                (
                    _to_xp_f64(
                        fcisolver.contract_2e(op_k, c0, cache.ncas, cache.nelecas, **_c2e_kw),
                        xp,
                    ).ravel()
                    if on_gpu
                    else _to_np_f64(
                        fcisolver.contract_2e(op_k, c0, cache.ncas, cache.nelecas, **_c2e_kw)
                    ).ravel()
                )
                for c0 in _ci0
            ]
        kci0 = [
            _to_xp_f64(kc0, xp) - xp.dot(_to_xp_f64(kc0, xp), _to_xp_f64(c0, xp)) * _to_xp_f64(c0, xp)
            for kc0, c0 in zip(kci0, _ci0)
        ]
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
            [xp.dot(_to_xp_f64(c1, xp), _to_xp_f64(c0, xp)) * 2.0 * wi for c1, c0, wi in zip(ci1_list, _ci0, _weights)],
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
        # Legacy Davidson/Newton callers still operate in NumPy, but GPU Krylov
        # paths (e.g. Z-vector GCROTMK/GMRES) can keep the Hessian matvec
        # fully device-resident. Only round-trip to host when the input vector
        # came from a NumPy caller.
        if on_gpu and _hop_on_gpu and (not bool(_input_is_gpu)):
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

    # Allow external callers (e.g. gradient driver) to explicitly release any
    # GPU-resident MO-JK precompute buffers captured by the matvec closure.
    try:
        setattr(_h_op_raw, "release_mo_jk", _release_mo_jk_precompute)
    except Exception:
        pass
    try:
        setattr(h_op, "release_mo_jk", _release_mo_jk_precompute)
    except Exception:
        pass

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
    ci_list: Sequence[Any],
    *,
    ref_list: Sequence[Any] | None = None,
    eps: float = 1e-12,
) -> list[Any]:
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

    xp, _ = _get_xp(*(ci_list[:1] if len(ci_list) else []))
    c_list = [xp.asarray(c, dtype=xp.float64).ravel() for c in ci_list]
    nroots = int(len(c_list))
    if nroots == 0:
        raise ValueError("empty CI list")
    if nroots == 1:
        c0 = c_list[0]
        nrm = _scalar_real_float(xp.linalg.norm(c0))
        return [c0 / nrm] if nrm > 0.0 else [c0]

    nci = int(c_list[0].size)
    if any(int(c.size) != nci for c in c_list):
        raise ValueError("inconsistent CI sizes across roots")

    cmat = xp.stack(c_list, axis=1)  # (nci,nroots)
    s = cmat.T @ cmat
    evals, evecs = xp.linalg.eigh(s)
    evals = xp.maximum(evals, float(eps))
    s_inv_sqrt = (evecs * (1.0 / xp.sqrt(evals))[None, :]) @ evecs.T
    q = cmat @ s_inv_sqrt

    if ref_list is not None:
        ref = [xp.asarray(c, dtype=xp.float64).ravel() for c in ref_list]
        if len(ref) == nroots and all(int(r.size) == nci for r in ref):
            for i in range(nroots):
                if _scalar_real_float(xp.dot(ref[i], q[:, i])) < 0.0:
                    q[:, i] *= -1.0

    return [xp.ascontiguousarray(q[:, i]) for i in range(nroots)]


def extract_rotation(
    casscf: Any,
    dr: Any,
    u: Any,
    ci0: Any,
    *,
    ci_update: str = "pyscf",
) -> tuple[Any, Any]:
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

    ci0_list = _as_ci_list(ci0)
    xp, _ = _get_xp(dr, u, *(ci0_list[:1]))

    dr = xp.asarray(dr, dtype=xp.float64).ravel()
    u = xp.asarray(u, dtype=xp.float64)

    nmo = int(casscf.mo_coeff.shape[1])
    frozen = getattr(casscf, "frozen", None)
    ngorb = int(np.count_nonzero(casscf.uniq_var_indices(nmo, casscf.ncore, casscf.ncas, frozen)))

    u = u @ _to_xp_f64(casscf.update_rotate_matrix(dr[:ngorb]), xp)

    p0 = int(ngorb)
    ci1_list: list[Any] = []
    for c0 in ci0_list:
        p1 = p0 + int(c0.size)
        d = xp.asarray(c0, dtype=xp.float64).ravel() + dr[p0:p1]
        nrm = _norm_f64(d)
        if nrm > 0.0:
            d = d / nrm
        ci1_list.append(xp.ascontiguousarray(d))
        p0 = p1

    if len(ci1_list) > 1:
        mode = str(ci_update).strip().lower()
        if mode not in ("pyscf", "orthonormalize"):
            raise ValueError("ci_update must be 'pyscf' or 'orthonormalize'")
        if mode == "orthonormalize":
            ci1_list = _orthonormalize_ci_columns(
                ci1_list,
                ref_list=ci0_list,
            )

    if not isinstance(ci0, (list, tuple)):
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
    xp, _ = _get_xp(g_all, mo, x0_guess)
    g_all = xp.asarray(g_all, dtype=xp.float64).ravel()

    frozen = getattr(casscf, "frozen", None)
    ngorb = int(np.count_nonzero(casscf.uniq_var_indices(nmo, casscf.ncore, casscf.ncas, frozen)))

    norm_gkf = norm_gall = _norm_f64(g_all)
    log.debug(
        "    |g|=%5.3g (%4.3g %4.3g) (keyframe)",
        norm_gall,
        _norm_f64(g_all[:ngorb]),
        _norm_f64(g_all[ngorb:]),
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

    def _step_maxabs(x: Any) -> tuple[float, float, float]:
        x_arr = xp.asarray(x, dtype=xp.float64).ravel()
        max_all = _scalar_real_float(xp.max(xp.abs(x_arr))) if x_arr.size else 0.0
        max_orb = _scalar_real_float(xp.max(xp.abs(x_arr[:ngorb]))) if ngorb > 0 else 0.0
        max_ci = _scalar_real_float(xp.max(xp.abs(x_arr[ngorb:]))) if ngorb < int(x_arr.size) else 0.0
        return max_orb, max_ci, max_all

    def _scale_to_caps(dxi: Any, hdxi: Any, cap_orb: float, cap_ci: float) -> tuple[Any, Any]:
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
            dxi = xp.asarray(dxi, dtype=xp.float64) * scale
            hdxi = xp.asarray(hdxi, dtype=xp.float64) * scale
        return dxi, hdxi

    def _mu_correct_hdxi(hdxi: Any, dxi: Any, muo: float, muc: float) -> Any:
        """Convert (H+mu)x to Hx without extra h_op calls."""
        hdxi = xp.asarray(hdxi, dtype=xp.float64).ravel().copy()
        dxi = xp.asarray(dxi, dtype=xp.float64).ravel()
        if muo != 0.0 and ngorb > 0:
            hdxi[:ngorb] -= float(muo) * dxi[:ngorb]
        if muc != 0.0 and ngorb < int(dxi.size):
            hdxi[ngorb:] -= float(muc) * dxi[ngorb:]
        return hdxi

    def _make_precond(hdiag: Any, level_shift: float, muo: float, muc: float) -> Callable[[Any, float], Any]:
        def _p(x: Any, e: float) -> Any:
            x_arr = xp.asarray(x, dtype=xp.float64).ravel()
            if callable(hdiag):
                # Callable preconditioners are assumed to already handle sign/shift robustly.
                out = xp.asarray(hdiag(x_arr, e - float(level_shift)), dtype=xp.float64).ravel()
                nrm = _norm_f64(out)
                if nrm > 0.0:
                    out *= 1.0 / nrm
                return out

            hdiagd = xp.asarray(hdiag, dtype=xp.float64).ravel() - (float(e) - float(level_shift))
            if muo != 0.0 and ngorb > 0:
                hdiagd[:ngorb] += float(muo)
            if muc != 0.0 and ngorb < int(hdiagd.size):
                hdiagd[ngorb:] += float(muc)

            # Sign-preserving floor for near-singular denominators.
            eps = 1e-8
            mask = xp.abs(hdiagd) < eps
            if _scalar_real_float(xp.any(mask)):
                hdiagd = hdiagd.copy()
                hdiagd[mask] = xp.copysign(eps, hdiagd[mask])
            out = x_arr / hdiagd
            nrm = _norm_f64(out)
            if nrm > 0.0:
                out *= 1.0 / nrm
            return out

        return _p

    def _make_h_op_mu(hop: Callable[[Any], Any], muo: float, muc: float) -> Callable[[Any], Any]:
        if muo == 0.0 and muc == 0.0:
            return hop

        def _hop_mu(x: Any) -> Any:
            x_arr = xp.asarray(x, dtype=xp.float64).ravel()
            y = xp.asarray(hop(x_arr), dtype=xp.float64).ravel()
            if muo != 0.0 and ngorb > 0:
                y[:ngorb] += float(muo) * x_arr[:ngorb]
            if muc != 0.0 and ngorb < int(x_arr.size):
                y[ngorb:] += float(muc) * x_arr[ngorb:]
            return y

        return _hop_mu

    stat = NewtonMicroStats(imic=0, tot_hop=0, tot_kf=1)
    stat.g_orb = _to_np_f64(g_all[:ngorb].copy()) if ngorb > 0 else None
    dr = xp.zeros_like(g_all)
    ikf = 0
    u = xp.eye(nmo, dtype=xp.float64)
    ci_kf: Any = ci0_use
    kf_trust_local = float(getattr(casscf, "kf_trust_region", 3.0))

    if x0_guess is None:
        x0_guess = g_all
    x0_guess = xp.asarray(x0_guess, dtype=xp.float64).ravel()
    g_op = lambda: g_all

    if norm_gall < float(conv_tol_grad) * 0.3:
        return u, ci_kf, norm_gall, stat, x0_guess

    last_dxi = xp.asarray(x0_guess, dtype=xp.float64).ravel()
    last_hdxi_true = xp.zeros_like(last_dxi)

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

        x0_use = xp.asarray(g_all, dtype=xp.float64).ravel() if n_retry > 0 else xp.asarray(last_dxi, dtype=xp.float64).ravel()
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

            norm_residual = _norm_f64(residual)
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

            dxi = xp.asarray(dxi, dtype=xp.float64).ravel()
            hdxi = xp.asarray(hdxi, dtype=xp.float64).ravel()
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
            g_trial = xp.asarray(g_all, dtype=xp.float64).ravel() + xp.asarray(hdxi_true, dtype=xp.float64).ravel()
            norm_g_trial = _norm_f64(g_trial)
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
            norm_dr = _norm_f64(dr)
            norm_gall = _norm_f64(g_all)
            norm_gorb = _norm_f64(g_all[:ngorb])
            norm_gci = _norm_f64(g_all[ngorb:])
            log.debug(
                "    imic %d(%d)  |g|=%3.2e (%2.1e %2.1e)  |dxi|=%3.2e max(o,c,a)=(%3.2e %3.2e %3.2e) |dr|=%3.2e  eig=%2.1e seig=%2.1e",
                stat.imic,
                hop_total,
                norm_gall,
                norm_gorb,
                norm_gci,
                _norm_f64(dxi),
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
                g_kf1 = xp.asarray(g_kf1, dtype=xp.float64).ravel()
                stat.tot_kf += 1

                norm_gkf1 = _norm_f64(g_kf1)
                norm_gorb = _norm_f64(g_kf1[:ngorb])
                norm_gci = _norm_f64(g_kf1[ngorb:])
                norm_dg = _norm_f64(g_kf1 - g_all)
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
                    dr[...] = 0.0
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
                norm_gall = _norm_f64(g_all)
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
            dci_kf = np.concatenate([(_to_np_f64(x) - _to_np_f64(y)).ravel() for x, y in zip(ci_kf, ci0_use)])
        else:
            dci_kf = _to_np_f64(ci_kf).ravel() - _to_np_f64(ci0_use).ravel()
    except Exception:
        dci_kf = np.zeros(1, dtype=np.float64)
    log.debug(
        "    tot inner=%d  |g|= %4.3g (%4.3g %4.3g) |u-1|= %4.3g  |dci|= %4.3g",
        stat.imic,
        norm_gall,
        _norm_f64(g_all[:ngorb]),
        _norm_f64(g_all[ngorb:]),
        _norm_f64(u - xp.eye(nmo, dtype=xp.float64)),
        float(np.linalg.norm(dci_kf)),
    )

    return u, ci_kf, norm_gkf, stat, xp.asarray(last_dxi, dtype=xp.float64).ravel()


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

    ppaa = getattr(eris, "ppaa", None)
    papa = getattr(eris, "papa", None)
    provider = getattr(eris, "eri_provider", None)
    C_act_ref = getattr(eris, "C_act", None)
    vhf_c = getattr(eris, "vhf_c", None)
    _provider_probe = None
    if provider is not None:
        _probe_fn = getattr(provider, "probe_array", None)
        if callable(_probe_fn):
            _provider_probe = _probe_fn()
    xp, _on_gpu = _get_xp(ppaa, papa, vhf_c, mo, C_act_ref, _provider_probe)
    # Build per-root RDMs.
    linkstr = _maybe_gen_linkstr(fcisolver, ncas, getattr(casscf, "nelecas"), False)
    if bool(_on_gpu):
        dm1s_l = []
        dm2s_l = []
        for c in ci0_list:
            dm1, dm2 = fcisolver.make_rdm12(
                c,
                ncas,
                getattr(casscf, "nelecas"),
                link_index=linkstr,
                rdm_backend="cuda",
                return_cupy=True,
                strict_gpu=True,
            )
            dm1s_l.append(xp.asarray(dm1, dtype=xp.float64))
            dm2s_l.append(xp.asarray(dm2, dtype=xp.float64))
        casdm1_r = xp.stack(dm1s_l, axis=0)
        casdm2_r = xp.stack(dm2s_l, axis=0)
    else:
        ci0_list_host = [_to_np_f64(c) for c in ci0_list]
        try:
            casdm1_r, casdm2_r = fcisolver.states_make_rdm12(
                ci0_list_host,
                ncas,
                getattr(casscf, "nelecas"),
                link_index=linkstr,
            )
            casdm1_r = np.asarray(casdm1_r, dtype=np.float64)
            casdm2_r = np.asarray(casdm2_r, dtype=np.float64)
        except AttributeError:
            dm1s_l = []
            dm2s_l = []
            for c in ci0_list_host:
                dm1, dm2 = fcisolver.make_rdm12(c, ncas, getattr(casscf, "nelecas"), link_index=linkstr)
                dm1s_l.append(np.asarray(dm1, dtype=np.float64))
                dm2s_l.append(np.asarray(dm2, dtype=np.float64))
            casdm1_r = np.asarray(dm1s_l, dtype=np.float64)
            casdm2_r = np.asarray(dm2s_l, dtype=np.float64)

    # State-averaged RDMs.
    w_xp = xp.asarray(w, dtype=xp.float64)
    casdm1 = xp.einsum("r,rpq->pq", w_xp, xp.asarray(casdm1_r, dtype=xp.float64), optimize=True)

    # Build per-root gpq and SA average.
    gpq = _build_gpq_per_root(
        casscf,
        mo,
        ci0_list,
        eris,
        strict_weights=bool(strict_weights),
        return_cupy=bool(_on_gpu),
    )
    g_orb_mat = xp.einsum("r,rpq->pq", w_xp, xp.asarray(gpq, dtype=xp.float64), optimize=True)
    g_orb = casscf.pack_uniq_var(g_orb_mat - g_orb_mat.T)
    g_orb = xp.asarray(g_orb, dtype=xp.float64).ravel() * 2.0

    casdm1_g = xp.asarray(casdm1, dtype=xp.float64)
    casdm2_g = xp.einsum("r,ruvwx->uvwx", w_xp, xp.asarray(casdm2_r, dtype=xp.float64), optimize=True)

    # Build cache intermediates needed for the orbital Hessian.
    vhf_c_xp = _to_xp_f64(vhf_c, xp)
    if provider is not None and C_act_ref is not None:
        mo_xp = _to_xp_f64(mo, xp)
        C_act_xp = _to_xp_f64(C_act_ref, xp)
        full_ppaa = xp.asarray(
            provider.build_pq_uv(mo_xp, C_act_xp),
            dtype=xp.float64,
        ).reshape(nmo, nmo, ncas, ncas)
        full_papa = xp.asarray(
            provider.build_pu_qv(mo_xp, C_act_xp),
            dtype=xp.float64,
        ).reshape(nmo, ncas, nmo, ncas)
        paaa = full_ppaa[:, ncore:nocc, :, :]
        occ_ppaa = full_ppaa[:nocc, :nocc, :, :]
        occ_papa = full_papa[:nocc, :, :nocc, :]

        D_act = C_act_xp @ casdm1_g @ C_act_xp.T
        Ja, Ka = provider.jk(D_act, want_J=True, want_K=True)
        if Ja is None or Ka is None:  # pragma: no cover
            raise RuntimeError("provider.jk returned None while J/K were requested")
        vhf_a = mo_xp.T @ (xp.asarray(Ja, dtype=xp.float64) - 0.5 * xp.asarray(Ka, dtype=xp.float64)) @ mo_xp
        vhf_ca = _to_np_f64(vhf_a + vhf_c_xp)

        arange_nocc = xp.arange(nocc)
        ppaa_diag = occ_ppaa[arange_nocc, arange_nocc]
        papa_diag = occ_papa[arange_nocc, :, arange_nocc]
        jkcaa_kernel = 6.0 * papa_diag - 2.0 * ppaa_diag
        jkcaa = _to_np_f64(xp.einsum("pik,ik->pi", jkcaa_kernel, casdm1_g, optimize=True))

        dm2tmp = casdm2_g.transpose(1, 2, 0, 3) + casdm2_g.transpose(0, 2, 1, 3)
        jtmp_full = xp.einsum("pquv,uvwx->pqwx", full_ppaa, casdm2_g, optimize=True)
        ktmp_full = xp.einsum("puqv,uvwx->pqwx", full_papa, dm2tmp, optimize=True)
        hdm2_xp = (jtmp_full + ktmp_full).transpose(0, 2, 1, 3)
        hdm2 = _to_np_f64(hdm2_xp)
    else:
        ppaa_arr = _to_xp_f64(ppaa, xp)
        papa_arr = _to_xp_f64(papa, xp)

        # vhf_a (SA-weighted), vhf_ca, jkcaa, hdm2
        vhf_a = xp.einsum("pquv,uv->pq", ppaa_arr, casdm1_g, optimize=True)
        vhf_a -= 0.5 * xp.einsum("puqv,uv->pq", papa_arr, casdm1_g, optimize=True)
        vhf_ca = _to_np_f64(vhf_a + vhf_c_xp)

        dm2tmp = casdm2_g.transpose(1, 2, 0, 3) + casdm2_g.transpose(0, 2, 1, 3)
        _ppaa_2d = ppaa_arr.reshape(nmo * nmo, ncas * ncas)
        _dm2_2d = casdm2_g.reshape(ncas * ncas, ncas * ncas)
        jtmp_full = (_ppaa_2d @ _dm2_2d).reshape(nmo, nmo, ncas, ncas)
        papa_t = papa_arr.transpose(0, 2, 1, 3)
        _papa_t_2d = papa_t.reshape(nmo * nmo, ncas * ncas)
        _dm2tmp_2d = dm2tmp.reshape(ncas * ncas, ncas * ncas)
        ktmp_full = (_papa_t_2d @ _dm2tmp_2d).reshape(nmo, nmo, ncas, ncas)
        hdm2 = _to_np_f64((jtmp_full + ktmp_full).transpose(0, 2, 1, 3))

        arange_nocc = xp.arange(nocc)
        ppaa_diag = ppaa_arr[arange_nocc, arange_nocc]
        papa_diag = papa_arr[arange_nocc, :, arange_nocc]
        jkcaa_kernel = 6.0 * papa_diag - 2.0 * ppaa_diag
        jkcaa = _to_np_f64(xp.einsum("pik,ik->pi", jkcaa_kernel, casdm1_g, optimize=True))

        paaa = ppaa_arr[:, ncore:nocc, :, :]

    hcore = casscf.get_hcore()
    h1e_mo = _to_np_f64((_to_xp_f64(mo, xp).T @ _to_xp_f64(hcore, xp)) @ _to_xp_f64(mo, xp))
    vhf_c_np = _to_np_f64(vhf_c_xp)

    # dm1_full
    dm1_full = np.zeros((nmo, nmo), dtype=np.float64)
    if ncore:
        idx_c = np.arange(ncore)
        dm1_full[idx_c, idx_c] = 2.0
    dm1_full[ncore:nocc, ncore:nocc] = _to_np_f64(casdm1)

    # ── Precompute MO-basis 3-index DF integrals for fast block-selective JK ──
    #
    # This is the dominant transient VRAM spike in orbital-only AH when it is
    # rebuilt repeatedly during rejected keyframe trust checks.  We make it
    # lazy and prefer packed (Qp) storage in MO-pair space.
    import os as _os_ghop

    _mojk_vram_debug = _env_bool_from_map(_os_ghop.environ, "ASUKA_MO_JK_VRAM_DEBUG", default=False)
    _mojk_vram_debug_fine = _env_bool_from_map(_os_ghop.environ, "ASUKA_MO_JK_VRAM_DEBUG_FINE", default=False)
    _canonical_mo_df_layout = str(_os_ghop.environ.get("ASUKA_MO_JK_CANONICAL_LAYOUT", "1")).strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
        "disable",
        "disabled",
    }
    _mojk_kblk_logged = {"done": False}

    _mojk_state: dict[str, Any] = {
        "ready": False,
        "use": False,
        "L_qp": None,          # (naux, ntri_mo) packed triangle in MO-pair space
        "L_t": None,           # (nmo, naux, nmo) legacy dense layout (fallback)
        "L_pq2d_act": None,    # optional legacy materialized (ncas*nmo, naux)
        "L_pq2d_core": None,   # optional legacy materialized (ncore*nmo, naux)
        # Optional dense caches derived from packed L_qp to avoid per-matvec Qp extraction.
        "L_core_rest_dense": None,  # (naux, ncore, nrest)
        "L_act_all_dense": None,    # (naux, ncas, nmo)
        # GPU-resident copies of h_op intermediates
        "h1e_mo": None,
        "dm1_full": None,
        "g_orb_sym": None,
        "vhf_ca": None,
        "vhf_c": None,
        "casdm1": None,
        "hdm2": None,
    }

    def _ensure_orb_mojk_precompute() -> None:
        if bool(_mojk_state["ready"]):
            return
        _mojk_state["ready"] = True

        if _os_ghop.environ.get("ASUKA_DISABLE_MO_JK", "0") == "1" or ncore <= 0 or _cp is None:
            return
        _df_B_raw = getattr(casscf, "df_B", None)
        if _df_B_raw is None:
            return

        try:
            _B_g = _cp.asarray(_df_B_raw, dtype=_cp.float64)
            _C_g = _cp.asarray(mo, dtype=_cp.float64)
            _nao_C = int(_C_g.shape[0])

            _allow_packed_mojk = str(_os_ghop.environ.get("ASUKA_MO_JK_ALLOW_PACKED_QP", "1")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            _use_qp_apply_kernel = str(_os_ghop.environ.get("ASUKA_MO_JK_QP_APPLY_KERNEL", "0")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            # Orbital-only AH matvec is extremely sensitive to the MO-JK storage layout.
            # Default to packed-Qp (lower VRAM, uses custom K kernels). Dense L_t can be
            # enabled explicitly via `ASUKA_MO_JK_ORB_STORAGE=lt` when desired.
            _orb_storage_policy = str(_os_ghop.environ.get("ASUKA_MO_JK_ORB_STORAGE", "qp")).strip().lower()

            _L_qp_gpu = None
            _L_t_gpu = None
            _L_core_rest_dense = None
            _L_act_all_dense = None

            # Packed Qp path (preferred for VRAM).
            if int(getattr(_B_g, "ndim", 0)) == 2 and bool(_allow_packed_mojk):
                from asuka.integrals.df_packed_s2 import (  # noqa: PLC0415
                    apply_Qp_to_C_block,
                    pack_Lf_block_to_Qp,
                    unpack_Qp_to_Qmn_block,
                )
                from asuka.integrals.tri_packed import ntri_from_nao  # noqa: PLC0415

                _naux_B, _ntri_B = map(int, _B_g.shape)
                _nao_B = int(_nao_C)
                if int(_ntri_B) == int(ntri_from_nao(int(_nao_B))):
                    _ntri_mo = int(ntri_from_nao(int(nmo)))
                    # Decide whether to store MO DF integrals as packed-Qp (low VRAM,
                    # slower matvec due to repeated block extraction) or dense L_t
                    # (higher VRAM, faster matvec).
                    _force_dense = str(_orb_storage_policy) in {"lt", "dense", "dense_lt", "dense-lt", "l_t"}
                    _force_packed = str(_orb_storage_policy) in {"qp", "packed", "packed_qp", "packed-qp"}
                    _prefer_dense = False
                    if bool(_force_dense):
                        _prefer_dense = True
                    elif bool(_force_packed):
                        _prefer_dense = False
                    else:
                        # Auto: allow dense only if it fits comfortably in current free VRAM
                        # (so we don't trigger multi-GB transient spikes on smaller GPUs).
                        try:
                            _free_b, _total_b = _cp.cuda.runtime.memGetInfo()
                            _cap_b = int(_total_b)
                            try:
                                _cap_gb = float(str(_os_ghop.environ.get("ASUKA_GPU_MEM_CAP_GB", "")).strip() or "0")
                            except Exception:
                                _cap_gb = 0.0
                            if float(_cap_gb) > 0.0:
                                _cap_b = int(min(float(_cap_b), float(_cap_gb) * 1e9))
                            _used_b = int(max(0, int(_total_b) - int(_free_b)))
                            _free_under_cap = int(max(0, int(_cap_b) - int(_used_b)))
                            _lt_bytes = int(nmo) * int(_naux_B) * int(nmo) * 8
                            _prefer_dense = bool(_lt_bytes <= int(_free_under_cap * 0.25))
                        except Exception:
                            _prefer_dense = False

                    if bool(_prefer_dense):
                        # Dense L_t: (nmo,naux,nmo). Reuse a persistent workspace to
                        # avoid repeated allocate/free spikes during keyframe refreshes.
                        _qblk = _choose_mojk_aux_qblk(
                            _cp,
                            _os_ghop.environ,
                            naux=int(_naux_B),
                            default_qblk=int(max(1, min(_naux_B, 64))),
                            bytes_per_q=int(
                                _estimate_mojk_precompute_bytes_per_q(
                                    mode="qp",
                                    nao=int(_nao_B),
                                    nmo=int(nmo),
                                    dtype_nbytes=8,
                                    store_packed=False,
                                    use_qp_kernel=bool(_use_qp_apply_kernel),
                                )
                            ),
                            label="orb_hop MO-JK precompute (L_t dense from Qp)",
                            debug=bool(_mojk_vram_debug),
                        )
                        _ws = getattr(casscf, "_mojk_orb_ws", None)
                        if not isinstance(_ws, dict):
                            _ws = {}
                            setattr(casscf, "_mojk_orb_ws", _ws)
                        _ws_Lt = _ws.get("L_t", None)
                        if (
                            _ws_Lt is None
                            or (not isinstance(_ws_Lt, _cp.ndarray))
                            or tuple(map(int, _ws_Lt.shape)) != (int(nmo), int(_naux_B), int(nmo))
                            or _ws_Lt.dtype != _cp.float64
                        ):
                            _ws_Lt = _cp.empty((int(nmo), int(_naux_B), int(nmo)), dtype=_cp.float64)
                            _ws["L_t"] = _ws_Lt
                        _L_t_gpu = _ws_Lt

                        if bool(_mojk_vram_debug):
                            _log_cuda_vram_snapshot(_cp, "orb_hop mojk precompute L_t(Qp) start", sync=False)
                        for _q0 in range(0, _naux_B, _qblk):
                            _q1 = min(_naux_B, _q0 + _qblk)
                            _qb = int(_q1 - _q0)
                            if _qb <= 0:
                                continue
                            if bool(_mojk_vram_debug_fine):
                                _log_cuda_vram_snapshot(
                                    _cp, f"orb_hop mojk L_t(Qp) q[{int(_q0)}:{int(_q1)}) start", sync=True
                                )
                            if _use_qp_apply_kernel:
                                _X_blk = apply_Qp_to_C_block(
                                    _B_g,
                                    _C_g,
                                    nao=int(_nao_B),
                                    q0=int(_q0),
                                    q_count=int(_qb),
                                )  # (qb,nao,nmo)
                            else:
                                _B_qmn = unpack_Qp_to_Qmn_block(_B_g, nao=int(_nao_B), q0=int(_q0), q_count=int(_qb))
                                # Avoid CuPy's batched-matmul workspace spikes by reshaping
                                # the (qb,nao,nao) block into a single 2D GEMM.
                                _X_f = _B_qmn.reshape(_qb * _nao_B, _nao_B) @ _C_g  # (qb*nao,nmo)
                                _X_blk = _X_f.reshape(_qb, _nao_B, int(nmo))  # (qb,nao,nmo)
                                del _X_f
                                del _B_qmn

                            _X_t_blk = _cp.ascontiguousarray(_X_blk.transpose(1, 0, 2))
                            _L_f_blk = _C_g.T @ _X_t_blk.reshape(_nao_B, _qb * nmo)  # (nmo, qb*nmo)
                            del _X_t_blk, _X_blk
                            _L_t_gpu[:, _q0:_q1, :] = _L_f_blk.reshape(nmo, _qb, nmo)
                            del _L_f_blk
                            if bool(_mojk_vram_debug_fine):
                                _log_cuda_vram_snapshot(
                                    _cp, f"orb_hop mojk L_t(Qp) q[{int(_q0)}:{int(_q1)}) done", sync=True
                                )
                        if bool(_mojk_vram_debug):
                            _log_cuda_vram_snapshot(_cp, "orb_hop mojk precompute L_t(Qp) done", sync=True)
                    else:
                        # Packed Qp in MO-pair space: (naux, ntri_mo)
                        _qblk = _choose_mojk_aux_qblk(
                            _cp,
                            _os_ghop.environ,
                            naux=int(_naux_B),
                            default_qblk=int(max(1, min(_naux_B, 32))),
                            bytes_per_q=int(
                                _estimate_mojk_precompute_bytes_per_q(
                                    mode="qp",
                                    nao=int(_nao_B),
                                    nmo=int(nmo),
                                    dtype_nbytes=8,
                                    store_packed=True,
                                    use_qp_kernel=bool(_use_qp_apply_kernel),
                                )
                            ),
                            label="orb_hop MO-JK precompute (Qp packed)",
                            debug=bool(_mojk_vram_debug),
                        )

                        _L_qp_gpu = _cp.empty((_naux_B, _ntri_mo), dtype=_cp.float64)
                        if bool(_mojk_vram_debug):
                            _log_cuda_vram_snapshot(_cp, "orb_hop mojk precompute Qp start", sync=False)
                        for _q0 in range(0, _naux_B, _qblk):
                            _q1 = min(_naux_B, _q0 + _qblk)
                            _qb = int(_q1 - _q0)
                            if _qb <= 0:
                                continue
                            if bool(_mojk_vram_debug_fine):
                                _log_cuda_vram_snapshot(_cp, f"orb_hop mojk Qp q[{int(_q0)}:{int(_q1)}) start", sync=True)
                            if _use_qp_apply_kernel:
                                _X_blk = apply_Qp_to_C_block(
                                    _B_g,
                                    _C_g,
                                    nao=int(_nao_B),
                                    q0=int(_q0),
                                    q_count=int(_qb),
                                )  # (qb,nao,nmo)
                            else:
                                _B_qmn = unpack_Qp_to_Qmn_block(_B_g, nao=int(_nao_B), q0=int(_q0), q_count=int(_qb))
                                # Avoid CuPy's batched-matmul workspace spikes by reshaping
                                # the (qb,nao,nao) block into a single 2D GEMM.
                                _X_f = _B_qmn.reshape(_qb * _nao_B, _nao_B) @ _C_g  # (qb*nao,nmo)
                                _X_blk = _X_f.reshape(_qb, _nao_B, int(nmo))  # (qb,nao,nmo)
                                del _X_f
                                del _B_qmn

                            # Build all q in one GEMM, then pack directly from the GEMM output
                            # to avoid materializing dense (q,nmo,nmo) buffers.
                            _X_t_blk = _cp.ascontiguousarray(_X_blk.transpose(1, 0, 2))
                            _L_f_blk = _C_g.T @ _X_t_blk.reshape(_nao_B, _qb * nmo)  # (nmo, qb*nmo)
                            del _X_t_blk
                            pack_Lf_block_to_Qp(
                                _L_f_blk,
                                _L_qp_gpu,
                                naux=int(_naux_B),
                                nao=int(nmo),
                                q0=int(_q0),
                                q_count=int(_qb),
                                threads=256,
                                sync=False,
                            )
                            del _L_f_blk, _X_blk
                            if bool(_mojk_vram_debug_fine):
                                _log_cuda_vram_snapshot(_cp, f"orb_hop mojk Qp q[{int(_q0)}:{int(_q1)}) done", sync=True)
                        if bool(_mojk_vram_debug):
                            _log_cuda_vram_snapshot(_cp, "orb_hop mojk precompute Qp done", sync=True)

                        # Optional dense caches for J-only contractions in packed-Qp matvec.
                        # This avoids repeated per-call `extract_Qp_rows_cols_block` work.
                        if int(ncore) > 0 and _env_bool_from_map(_os_ghop.environ, "ASUKA_MO_JK_PACKED_J_CACHE", default=False):
                            try:
                                from asuka.integrals.df_packed_s2 import extract_Qp_rows_cols_block  # noqa: PLC0415

                                _nrest = int(nmo - ncore)
                                _ws = getattr(casscf, "_mojk_orb_ws", None)
                                if not isinstance(_ws, dict):
                                    _ws = {}
                                    setattr(casscf, "_mojk_orb_ws", _ws)

                                _core_rest_out = _ws.get("L_core_rest_dense", None)
                                if (
                                    _core_rest_out is None
                                    or (not isinstance(_core_rest_out, _cp.ndarray))
                                    or tuple(map(int, _core_rest_out.shape)) != (int(_naux_B), int(ncore), int(_nrest))
                                    or _core_rest_out.dtype != _cp.float64
                                ):
                                    _core_rest_out = _cp.empty((int(_naux_B), int(ncore), int(_nrest)), dtype=_cp.float64)
                                    _ws["L_core_rest_dense"] = _core_rest_out
                                _L_core_rest_dense = extract_Qp_rows_cols_block(
                                    _L_qp_gpu,
                                    nao=int(nmo),
                                    q0=0,
                                    q_count=int(_naux_B),
                                    row0=0,
                                    row_count=int(ncore),
                                    col0=int(ncore),
                                    col_count=int(_nrest),
                                    out=_core_rest_out,
                                )

                                _act_all_out = _ws.get("L_act_all_dense", None)
                                if (
                                    _act_all_out is None
                                    or (not isinstance(_act_all_out, _cp.ndarray))
                                    or tuple(map(int, _act_all_out.shape)) != (int(_naux_B), int(ncas), int(nmo))
                                    or _act_all_out.dtype != _cp.float64
                                ):
                                    _act_all_out = _cp.empty((int(_naux_B), int(ncas), int(nmo)), dtype=_cp.float64)
                                    _ws["L_act_all_dense"] = _act_all_out
                                _L_act_all_dense = extract_Qp_rows_cols_block(
                                    _L_qp_gpu,
                                    nao=int(nmo),
                                    q0=0,
                                    q_count=int(_naux_B),
                                    row0=int(ncore),
                                    row_count=int(ncas),
                                    col0=0,
                                    col_count=int(nmo),
                                    out=_act_all_out,
                                )
                            except Exception:
                                _L_core_rest_dense = None
                                _L_act_all_dense = None

            # Dense mnQ path (fallback).
            if _L_qp_gpu is None and int(getattr(_B_g, "ndim", 0)) == 3 and int(_B_g.shape[0]) == int(_B_g.shape[1]):
                _nao_B = int(_B_g.shape[0])
                _naux_B = int(_B_g.shape[2])
                _qblk = _choose_mojk_aux_qblk(
                    _cp,
                    _os_ghop.environ,
                    naux=int(_naux_B),
                    default_qblk=int(max(1, min(_naux_B, 96))),
                    bytes_per_q=int(
                        _estimate_mojk_precompute_bytes_per_q(
                            mode="mnq",
                            nao=int(_nao_B),
                            nmo=int(nmo),
                            dtype_nbytes=8,
                            store_packed=False,
                            use_qp_kernel=True,
                        )
                    ),
                    label="orb_hop MO-JK precompute (mnQ)",
                    debug=bool(_mojk_vram_debug),
                )

                _L_t_gpu = _cp.empty((nmo, _naux_B, nmo), dtype=_cp.float64)
                if bool(_mojk_vram_debug):
                    _log_cuda_vram_snapshot(_cp, "orb_hop mojk precompute mnQ start", sync=False)
                for _q0 in range(0, _naux_B, _qblk):
                    _q1 = min(_naux_B, _q0 + _qblk)
                    _qb = int(_q1 - _q0)
                    if _qb <= 0:
                        continue
                    if bool(_mojk_vram_debug_fine):
                        _log_cuda_vram_snapshot(_cp, f"orb_hop mojk mnQ q[{int(_q0)}:{int(_q1)}) start", sync=True)
                    _B_blk = _B_g[:, :, _q0:_q1]
                    _H_blk = (_C_g.T @ _B_blk.reshape(_nao_B, _nao_B * _qb)).reshape(nmo, _nao_B, _qb)
                    _H_t_blk = _cp.ascontiguousarray(_H_blk.transpose(0, 2, 1))  # (nmo,qb,nao)
                    _L_blk = (_H_t_blk.reshape(nmo * _qb, _nao_B) @ _C_g).reshape(nmo, _qb, nmo)
                    _L_t_gpu[:, _q0:_q1, :] = _L_blk
                    del _B_blk, _H_blk, _H_t_blk, _L_blk
                    if bool(_mojk_vram_debug_fine):
                        _log_cuda_vram_snapshot(_cp, f"orb_hop mojk mnQ q[{int(_q0)}:{int(_q1)}) done", sync=True)
                _L_t_gpu = _cp.ascontiguousarray(_L_t_gpu)
                if bool(_mojk_vram_debug):
                    _log_cuda_vram_snapshot(_cp, "orb_hop mojk precompute mnQ done", sync=True)

            del _B_g, _C_g

            if _L_qp_gpu is None and _L_t_gpu is None:
                return

            _mojk_state["L_qp"] = _L_qp_gpu
            _mojk_state["L_t"] = _L_t_gpu
            _mojk_state["L_core_rest_dense"] = _L_core_rest_dense
            _mojk_state["L_act_all_dense"] = _L_act_all_dense
            if _L_t_gpu is not None and (not _canonical_mo_df_layout):
                _mojk_state["L_pq2d_act"] = _cp.ascontiguousarray(
                    _L_t_gpu[ncore:nocc].transpose(0, 2, 1).reshape(ncas * nmo, _L_t_gpu.shape[1])
                )
                _mojk_state["L_pq2d_core"] = _cp.ascontiguousarray(
                    _L_t_gpu[:ncore].transpose(0, 2, 1).reshape(ncore * nmo, _L_t_gpu.shape[1])
                )

            # GPU-resident copies of h_op intermediates (avoids per-call CPU→GPU transfers)
            _mojk_state["h1e_mo"] = _cp.asarray(h1e_mo, dtype=_cp.float64)
            _mojk_state["dm1_full"] = _cp.asarray(dm1_full, dtype=_cp.float64)
            _mojk_state["g_orb_sym"] = _cp.asarray(g_orb_mat + g_orb_mat.T, dtype=_cp.float64)
            _mojk_state["vhf_ca"] = _cp.asarray(vhf_ca, dtype=_cp.float64)
            _mojk_state["vhf_c"] = _cp.asarray(vhf_c_np, dtype=_cp.float64)
            _mojk_state["casdm1"] = _cp.asarray(casdm1, dtype=_cp.float64)
            _mojk_state["hdm2"] = _cp.asarray(hdm2, dtype=_cp.float64)

            _mojk_state["use"] = True
        except Exception:
            _mojk_state["use"] = False

    # Orbital diagonal Hessian (PySCF Parts 7-6).
    orb_xp = xp if bool(_on_gpu) else np
    h1e_mo_diag = _to_xp_f64(h1e_mo, orb_xp)
    dm1_full_diag = _to_xp_f64(dm1_full, orb_xp)
    vhf_ca_diag = _to_xp_f64(vhf_ca, orb_xp)
    vhf_c_diag = _to_xp_f64(vhf_c_np, orb_xp)
    casdm1_diag = _to_xp_f64(casdm1, orb_xp)
    jkcaa_diag = _to_xp_f64(jkcaa, orb_xp)
    hdm2_diag = _to_xp_f64(hdm2, orb_xp)
    h_diag = orb_xp.einsum("ii,jj->ij", h1e_mo_diag, dm1_full_diag) - h1e_mo_diag * dm1_full_diag
    h_diag = h_diag + h_diag.T
    g_diag = orb_xp.einsum("r,rpp->p", orb_xp.asarray(w, dtype=orb_xp.float64), orb_xp.asarray(gpq, dtype=orb_xp.float64), optimize=True)
    h_diag -= g_diag + g_diag.reshape(-1, 1)
    idx = orb_xp.arange(nmo)
    h_diag[idx, idx] += g_diag * 2.0
    v_diag = orb_xp.diag(vhf_ca_diag)
    h_diag[:, :ncore] += v_diag.reshape(-1, 1) * 2.0
    h_diag[:ncore] += v_diag * 2.0
    if ncore:
        idxc = orb_xp.arange(ncore)
        h_diag[idxc, idxc] -= v_diag[:ncore] * 4.0
    tmp_d = orb_xp.einsum("ii,jj->ij", vhf_c_diag, casdm1_diag, optimize=True)
    h_diag[:, ncore:nocc] += tmp_d
    h_diag[ncore:nocc, :] += tmp_d.T
    tmp2_d = -vhf_c_diag[ncore:nocc, ncore:nocc] * casdm1_diag
    h_diag[ncore:nocc, ncore:nocc] += tmp2_d + tmp2_d.T
    tmp3_d = 6.0 * _to_xp_f64(getattr(eris, "k_pc"), orb_xp) - 2.0 * _to_xp_f64(getattr(eris, "j_pc"), orb_xp)
    h_diag[ncore:, :ncore] += tmp3_d[ncore:]
    h_diag[:ncore, ncore:] += tmp3_d[ncore:].T
    h_diag[:nocc, ncore:nocc] -= jkcaa_diag
    h_diag[ncore:nocc, :nocc] -= jkcaa_diag.T
    v_diag2 = orb_xp.einsum("ijij->ij", hdm2_diag, optimize=True)
    h_diag[ncore:nocc, :] += v_diag2.T
    h_diag[:, ncore:nocc] += v_diag2
    h_diag_orb = casscf.pack_uniq_var(h_diag)
    if bool(_on_gpu):
        h_diag_orb = orb_xp.asarray(h_diag_orb, dtype=orb_xp.float64).ravel() * 2.0
    else:
        h_diag_orb = np.asarray(h_diag_orb, dtype=np.float64).ravel() * 2.0

    # Orbital-only Hessian-vector product.
    def _h_op_orb(x_packed: Any) -> Any:
        _xp_in, _x_on_gpu = _get_xp(x_packed)
        use_gpu = bool(_x_on_gpu or _on_gpu) and _cp is not None
        if bool(use_gpu):
            _xp_h = _cp
            x_packed_dev = _xp_h.asarray(x_packed, dtype=_xp_h.float64).ravel()
            x1 = casscf.unpack_uniq_var(x_packed_dev)
        else:
            x_packed = np.asarray(x_packed, dtype=np.float64).ravel()
            x1 = casscf.unpack_uniq_var(x_packed)

        _ensure_orb_mojk_precompute()
        if bool(use_gpu) and bool(_mojk_state.get("use", False)):
            # ── Full GPU path: matmuls + block-selective MO-basis JK ──
            _xp = _cp
            x1_g = _xp.asarray(x1, dtype=_xp.float64)

            _h1e_mo_g = _mojk_state["h1e_mo"]
            _dm1_full_g = _mojk_state["dm1_full"]
            _g_orb_sym_g = _mojk_state["g_orb_sym"]
            _vhf_ca_g = _mojk_state["vhf_ca"]
            _vhf_c_g = _mojk_state["vhf_c"]
            _casdm1_hop_g = _mojk_state["casdm1"]
            _hdm2_g = _mojk_state["hdm2"]

            # Orbital Hessian assembly (GPU)
            x2_g = (_h1e_mo_g @ x1_g) @ _dm1_full_g
            x2_g -= _g_orb_sym_g @ x1_g * 0.5
            if ncore:
                x2_g[:ncore] += (x1_g[:ncore, ncore:] @ _vhf_ca_g[ncore:]) * 2.0
            x2_g[ncore:nocc] += (_casdm1_hop_g @ x1_g[ncore:nocc]) @ _vhf_c_g
            x2_g[:, ncore:nocc] += _xp.einsum("purv,rv->pu", _hdm2_g, x1_g[:, ncore:nocc], optimize=True)

            # Block-selective MO-basis JK
            if ncore > 0:
                # dm3_MO: symmetric core↔rest block of x1
                dm3 = _xp.zeros((nmo, nmo), dtype=_xp.float64)
                dm3[:ncore, ncore:] = x1_g[:ncore, ncore:]
                dm3[ncore:, :ncore] = x1_g[:ncore, ncore:].T

                # dm_total = 2*dm3 + dm4
                dm4_h = _xp.zeros((nmo, nmo), dtype=_xp.float64)
                dm4_h[ncore:nocc, :] = _casdm1_hop_g @ x1_g[ncore:nocc, :]
                dm_total = dm3 * 2.0 + dm4_h + dm4_h.T

                _L_qp_gpu = _mojk_state.get("L_qp", None)
                if _L_qp_gpu is not None:
                    from asuka.hf import df_scf as _df_scf  # noqa: PLC0415
                    from asuka.integrals.df_packed_s2 import extract_Qp_rows_cols_block  # noqa: PLC0415

                    _naux_L = int(_L_qp_gpu.shape[0])
                    try:
                        _k_qblk = int(str(_os_ghop.environ.get("ASUKA_MO_JK_K_QBLOCK_PACKED", "")).strip() or "0")
                    except Exception:
                        _k_qblk = 0
                    if _k_qblk <= 0:
                        try:
                            _k_qblk = int(str(_os_ghop.environ.get("ASUKA_DF_JK_K_QBLOCK_PACKED", "")).strip() or "0")
                        except Exception:
                            _k_qblk = 0
                    if _k_qblk <= 0:
                        _k_qblk = 128
                    _k_qblk = int(max(1, min(_naux_L, _k_qblk)))

                    _k_cblk = _choose_mojk_colblk(
                        _xp,
                        _os_ghop.environ,
                        qblk=int(_k_qblk),
                        nao=int(nmo),
                        ncol_total=int(nmo),
                        default_colblk=256,
                        dtype_nbytes=8,
                        label="orb_hop packed-K",
                        debug=bool(_mojk_vram_debug and not bool(_mojk_kblk_logged["done"])),
                    )
                    _mojk_kblk_logged["done"] = True

                    # J blocks (avoid full J build; exploit dm3/dm4 structure like the dense-L_t path).
                    _nrest = int(nmo - ncore)
                    _x_core_rest = x1_g[:ncore, ncore:]  # (ncore, nrest)
                    _dm4_act = _casdm1_hop_g @ x1_g[ncore:nocc, :]  # (ncas, nmo)

                    _L_core_rest_dense = _mojk_state.get("L_core_rest_dense", None)
                    _L_act_all_dense = _mojk_state.get("L_act_all_dense", None)
                    if _L_core_rest_dense is not None and _L_act_all_dense is not None:
                        # Cached dense blocks: J is just a few contractions and avoids per-call Qp extraction.
                        _Lcr2d = _L_core_rest_dense.reshape(int(_naux_L), int(ncore * _nrest))
                        _Lact2d = _L_act_all_dense.reshape(int(_naux_L), int(ncas * nmo))
                        _rho0 = 2.0 * (_Lcr2d @ _x_core_rest.ravel())
                        _J0_act = (_rho0 @ _Lact2d).reshape(int(ncas), int(nmo))
                        _rho_dm4 = 2.0 * (_Lact2d @ _dm4_act.ravel())
                        _rho1 = _rho0 * 2.0 + _rho_dm4
                        _J1_core_rest = (_rho1 @ _Lcr2d).reshape(int(ncore), int(_nrest))
                        del _Lcr2d, _Lact2d, _rho0, _rho_dm4, _rho1
                    else:
                        _J0_act = _xp.zeros((ncas, nmo), dtype=_xp.float64)
                        _J1_core_rest = _xp.zeros((ncore, _nrest), dtype=_xp.float64)
                        for _q0 in range(0, _naux_L, _k_qblk):
                            _q1 = min(_naux_L, _q0 + _k_qblk)
                            _qb = int(_q1 - _q0)
                            if _qb <= 0:
                                continue

                            # rho0_Q = 2 * sum_{i in core, a in rest} L_{iQa} * x1[i,a]
                            # Use rest-core orientation (a,i) from packed storage: L_{iQa} == L_{aQi}.
                            _L_rest_core = extract_Qp_rows_cols_block(
                                _L_qp_gpu,
                                nao=int(nmo),
                                q0=int(_q0),
                                q_count=int(_qb),
                                row0=int(ncore),
                                row_count=int(_nrest),
                                col0=0,
                                col_count=int(ncore),
                            )
                            _rho0 = 2.0 * _xp.einsum("qai,ia->q", _L_rest_core, _x_core_rest, optimize=True)
                            del _L_rest_core

                            # Active rows for J0 and dm4 term.
                            _L_act_all = extract_Qp_rows_cols_block(
                                _L_qp_gpu,
                                nao=int(nmo),
                                q0=int(_q0),
                                q_count=int(_qb),
                                row0=int(ncore),
                                row_count=int(ncas),
                                col0=0,
                                col_count=int(nmo),
                            )
                            _J0_act += _xp.einsum("q,qpm->pm", _rho0, _L_act_all, optimize=True)
                            _rho_dm4 = 2.0 * _xp.einsum("qpm,pm->q", _L_act_all, _dm4_act, optimize=True)
                            del _L_act_all

                            # rho1 = rho(dm_total) = 2*rho0 + rho_dm4
                            _rho1 = _rho0 * 2.0 + _rho_dm4
                            del _rho0, _rho_dm4

                            # Core rows, rest cols for J1.
                            _L_core_rest = extract_Qp_rows_cols_block(
                                _L_qp_gpu,
                                nao=int(nmo),
                                q0=int(_q0),
                                q_count=int(_qb),
                                row0=0,
                                row_count=int(ncore),
                                col0=int(ncore),
                                col_count=int(_nrest),
                            )
                            _J1_core_rest += _xp.einsum("q,qim->im", _rho1, _L_core_rest, optimize=True)
                            del _L_core_rest, _rho1

                    _K0_act = _df_scf._df_K_qblocked_Qp_rows_cols(  # noqa: SLF001
                        _L_qp_gpu,
                        dm3,
                        nao=int(nmo),
                        row0=int(ncore),
                        row_count=int(ncas),
                        col0=0,
                        col_count=int(nmo),
                        q_block=int(_k_qblk),
                        col_block=int(_k_cblk),
                    )
                    _K1_core_rest = _df_scf._df_K_qblocked_Qp_rows_cols(  # noqa: SLF001
                        _L_qp_gpu,
                        dm_total,
                        nao=int(nmo),
                        row0=0,
                        row_count=int(ncore),
                        col0=int(ncore),
                        col_count=int(nmo - ncore),
                        q_block=int(_k_qblk),
                        col_block=int(_k_cblk),
                    )
                    x2_g[ncore:nocc] += _casdm1_hop_g @ (_J0_act * 2.0 - _K0_act)
                    x2_g[:ncore, ncore:] += (_J1_core_rest * 2.0 - _K1_core_rest)
                else:
                    # Legacy dense L_t path (fastest, higher VRAM). Kept as fallback.
                    _L_t_gpu = _mojk_state.get("L_t", None)
                    if _L_t_gpu is not None:
                        _naux = int(_L_t_gpu.shape[1])
                        L_t_act_flat = _L_t_gpu[ncore:nocc].reshape(ncas * _naux, nmo)
                        L_t_core_flat = _L_t_gpu[:ncore].reshape(ncore * _naux, nmo)

                        # K Step 1: separate small GEMMs for each density
                        LDM0 = (L_t_act_flat @ dm3).reshape(ncas, _naux * nmo)
                        LDM1 = (L_t_core_flat @ dm_total).reshape(ncore, _naux * nmo)

                        try:
                            _k_accum_qblk = int(str(_os_ghop.environ.get("ASUKA_MO_JK_K_ACCUM_QBLOCK", "")).strip() or "0")
                        except Exception:
                            _k_accum_qblk = 0
                        if _k_accum_qblk <= 0:
                            _k_accum_qblk = int(max(1, min(_naux, 64)))
                        else:
                            _k_accum_qblk = int(max(1, min(_naux, _k_accum_qblk)))

                        K0_act = _xp.zeros((ncas, nmo), dtype=LDM0.dtype)
                        K1_core = _xp.zeros((ncore, nmo), dtype=LDM1.dtype)
                        for _q0 in range(0, _naux, _k_accum_qblk):
                            _q1 = min(_naux, _q0 + _k_accum_qblk)
                            _qb = int(_q1 - _q0)
                            if _qb <= 0:
                                continue
                            _off0 = int(_q0 * nmo)
                            _off1 = int(_q1 * nmo)
                            _L_blk_t = _L_t_gpu[:, _q0:_q1, :].reshape(nmo, _qb * nmo).T
                            K0_act += LDM0[:, _off0:_off1] @ _L_blk_t
                            K1_core += LDM1[:, _off0:_off1] @ _L_blk_t

                        # J0: exploit sparse dm3 (only core↔rest nonzero)
                        rho0 = 2.0 * _xp.einsum(
                            "iQa,ia->Q", _L_t_gpu[:ncore, :, ncore:], x1_g[:ncore, ncore:], optimize=True
                        )
                        if _canonical_mo_df_layout:
                            J0_act = _xp.einsum("pQq,Q->pq", _L_t_gpu[ncore:nocc], rho0, optimize=True)
                        else:
                            _L_pq2d_act_gpu = _mojk_state.get("L_pq2d_act", None)
                            J0_act = (_L_pq2d_act_gpu @ rho0).reshape(ncas, nmo)

                        # J1: rho1 = 2*rho0 + rho_dm4
                        dm4_act = _casdm1_hop_g @ x1_g[ncore:nocc]
                        if _canonical_mo_df_layout:
                            rho_dm4 = 2.0 * _xp.einsum("pQq,pq->Q", _L_t_gpu[ncore:nocc], dm4_act, optimize=True)
                        else:
                            _L_pq2d_act_gpu = _mojk_state.get("L_pq2d_act", None)
                            rho_dm4 = 2.0 * (_L_pq2d_act_gpu.T @ dm4_act.ravel())
                        rho1 = 2.0 * rho0 + rho_dm4
                        if _canonical_mo_df_layout:
                            J1_core = _xp.einsum("pQq,Q->pq", _L_t_gpu[:ncore], rho1, optimize=True)
                        else:
                            _L_pq2d_core_gpu = _mojk_state.get("L_pq2d_core", None)
                            J1_core = (_L_pq2d_core_gpu @ rho1).reshape(ncore, nmo)

                    x2_g[ncore:nocc] += _casdm1_hop_g @ (J0_act * 2.0 - K0_act)
                    x2_g[:ncore, ncore:] += (J1_core * 2.0 - K1_core)[:, ncore:]

            x2_g = x2_g - x2_g.T
            out = _xp.asarray(casscf.pack_uniq_var(x2_g), dtype=_xp.float64).ravel() * 2.0
            if not bool(_x_on_gpu):
                out = out.get()
            return out
        else:
            if bool(use_gpu):
                _xp = _cp
                x1_g = _xp.asarray(x1, dtype=_xp.float64)
                _h1e_mo = _xp.asarray(h1e_mo, dtype=_xp.float64)
                _dm1_full = _xp.asarray(dm1_full, dtype=_xp.float64)
                _g_orb_sym = _xp.asarray(g_orb_mat + g_orb_mat.T, dtype=_xp.float64)
                _vhf_ca = _xp.asarray(vhf_ca, dtype=_xp.float64)
                _casdm1 = _xp.asarray(casdm1, dtype=_xp.float64)
                _vhf_c = _xp.asarray(vhf_c_np, dtype=_xp.float64)
                _hdm2 = _xp.asarray(hdm2, dtype=_xp.float64)
                x2 = (_h1e_mo @ x1_g) @ _dm1_full
                x2 -= _g_orb_sym @ x1_g * 0.5
                if ncore:
                    x2[:ncore] += (x1_g[:ncore, ncore:] @ _vhf_ca[ncore:]) * 2.0
                x2[ncore:nocc] += (_casdm1 @ x1_g[ncore:nocc]) @ _vhf_c
                x2[:, ncore:nocc] += _xp.einsum("purv,rv->pu", _hdm2, x1_g[:, ncore:nocc], optimize=True)

                if ncore > 0:
                    va, vc = casscf.update_jk_in_ah(mo, x1_g, _casdm1, eris)
                    x2[ncore:nocc] += _to_xp_f64(va, _xp)
                    x2[:ncore, ncore:] += _to_xp_f64(vc, _xp)

                x2 = x2 - x2.T
                return _xp.asarray(casscf.pack_uniq_var(x2), dtype=_xp.float64).ravel() * 2.0
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

    def _gorb_update(u_rot: Any, ci_new: Any) -> tuple[Any, Callable, Any]:
        _xp_u, _ = _get_xp(mo, u_rot)
        mo_new = _to_xp_f64(mo, _xp_u) @ _to_xp_f64(u_rot, _xp_u)
        return gen_g_hop_orbital(casscf, mo_new, ci_new, eris, weights=weights, strict_weights=strict_weights)[:3]

    return g_orb, _h_op_orb, h_diag_orb, _gorb_update


def orbital_hessian_action_matrix(
    casscf: Any,
    mo: np.ndarray,
    ci0: Any,
    eris: Any,
    x_packed: np.ndarray,
    *,
    weights: Sequence[float] | None = None,
    strict_weights: bool = False,
    antisymmetrize: bool = True,
) -> np.ndarray:
    """Return the orbital-only Hessian action matrix before packing.

    This mirrors the CPU path inside :func:`gen_g_hop_orbital`. When
    ``antisymmetrize=False``, the returned matrix is the pre-antisymmetrized
    ``x2`` object before ``x2 - x2.T`` and before ``pack_uniq_var`` scaling.
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

    casdm1 = np.einsum("r,rpq->pq", w, casdm1_r, optimize=True)
    gpq = _build_gpq_per_root(casscf, mo, ci0_list, eris, strict_weights=bool(strict_weights))
    g_orb_mat = np.einsum("r,rpq->pq", w, gpq, optimize=True)

    ppaa = getattr(eris, "ppaa", None)
    papa = getattr(eris, "papa", None)
    vhf_c = getattr(eris, "vhf_c", None)
    xp, _on_gpu = _get_xp(ppaa, papa)
    ppaa_arr = _to_xp_f64(ppaa, xp)
    papa_arr = _to_xp_f64(papa, xp)
    casdm1_g = xp.asarray(casdm1, dtype=xp.float64)
    casdm2_g = xp.asarray(np.einsum("r,ruvwx->uvwx", w, casdm2_r, optimize=True), dtype=xp.float64)

    vhf_a = xp.einsum("pquv,uv->pq", ppaa_arr, casdm1_g, optimize=True)
    vhf_a -= 0.5 * xp.einsum("puqv,uv->pq", papa_arr, casdm1_g, optimize=True)
    vhf_c_xp = _to_xp_f64(vhf_c, xp)
    vhf_ca = _to_np_f64(vhf_a) + _to_np_f64(vhf_c_xp)

    dm2tmp = casdm2_g.transpose(1, 2, 0, 3) + casdm2_g.transpose(0, 2, 1, 3)
    _ppaa_2d = ppaa_arr.reshape(nmo * nmo, ncas * ncas)
    _dm2_2d = casdm2_g.reshape(ncas * ncas, ncas * ncas)
    jtmp_full = (_ppaa_2d @ _dm2_2d).reshape(nmo, nmo, ncas, ncas)
    papa_t = papa_arr.transpose(0, 2, 1, 3)
    _papa_t_2d = papa_t.reshape(nmo * nmo, ncas * ncas)
    _dm2tmp_2d = dm2tmp.reshape(ncas * ncas, ncas * ncas)
    ktmp_full = (_papa_t_2d @ _dm2tmp_2d).reshape(nmo, nmo, ncas, ncas)
    hdm2 = _to_np_f64((jtmp_full + ktmp_full).transpose(0, 2, 1, 3))

    hcore = casscf.get_hcore()
    h1e_mo = _to_np_f64((_to_xp_f64(mo, xp).T @ _to_xp_f64(hcore, xp)) @ _to_xp_f64(mo, xp))
    vhf_c_np = _to_np_f64(vhf_c_xp)

    dm1_full = np.zeros((nmo, nmo), dtype=np.float64)
    if ncore:
        idx_c = np.arange(ncore)
        dm1_full[idx_c, idx_c] = 2.0
    dm1_full[ncore:nocc, ncore:nocc] = casdm1

    x1 = casscf.unpack_uniq_var(np.asarray(x_packed, dtype=np.float64).ravel())
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
    if bool(antisymmetrize):
        x2 = x2 - x2.T
    return np.asarray(x2, dtype=np.float64)


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
    force_cpu = bool(getattr(casscf, "_asuka_force_cpu", False))
    if force_cpu:
        _h_op_orb = h_op_orb
        _gorb_update = gorb_update

        def h_op_orb(x: Any) -> np.ndarray:
            return _to_np_f64(_h_op_orb(_to_np_f64(x))).ravel()

        def gorb_update(u_rot: Any, ci_new: Any) -> tuple[np.ndarray, Any, np.ndarray]:
            g_kf, h_op_kf, h_diag_kf = _gorb_update(_to_np_f64(u_rot), ci_new)

            def _h_op_kf_cpu(x: Any) -> np.ndarray:
                return _to_np_f64(h_op_kf(_to_np_f64(x))).ravel()

            return _to_np_f64(g_kf).ravel(), _h_op_kf_cpu, _to_np_f64(h_diag_kf).ravel()

        g_orb = _to_np_f64(g_orb).ravel()
        h_diag_orb = _to_np_f64(h_diag_orb).ravel()

    orb_xp, _ = _get_xp(g_orb)
    norm_gorb = _norm_f64(g_orb)
    norm_gkf = norm_gorb

    stat = OrbitalMicroStats(imic=0, tot_hop=0, tot_kf=1, norm_gorb=norm_gorb)

    if norm_gorb < 1e-8:
        return np.eye(nmo, dtype=np.float64), norm_gorb, stat

    if force_cpu:
        u = np.eye(nmo, dtype=np.float64)
    else:
        u = orb_xp.eye(nmo, dtype=orb_xp.float64)
    dr = orb_xp.zeros_like(g_orb)
    x0_guess = orb_xp.asarray(g_orb, dtype=orb_xp.float64).copy()

    def _make_precond(hdiag: np.ndarray, ls: float) -> Callable[[np.ndarray, float], np.ndarray]:
        def _p(x: np.ndarray, e: float) -> np.ndarray:
            x = _to_np_f64(x).ravel()
            hd = _to_np_f64(hdiag).ravel() - (float(e) - float(ls))
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

            norm_residual = _norm_f64(residual)
            accept_step = (
                bool(ah_conv)
                or int(ihop) == int(ah_max_cycle)
                or ((norm_residual < ah_start_tol) and (int(ihop) >= ah_start_cycle))
                or (float(seig) < ah_lindep)
            )
            if not accept_step:
                continue

            if force_cpu:
                dxi = _to_np_f64(dxi).ravel()
                hdxi = _to_np_f64(hdxi).ravel()
            else:
                dxi = _to_xp_f64(dxi, orb_xp).ravel()
                hdxi = _to_xp_f64(hdxi, orb_xp).ravel()

            # Clip step to max_stepsize.
            max_abs = _scalar_real_float(orb_xp.max(orb_xp.abs(dxi))) if int(dxi.size) > 0 else 0.0
            if max_abs > max_stepsize:
                scale = max_stepsize / max_abs
                dxi = dxi * scale
                hdxi = hdxi * scale
                log.debug1("Scale orbital AH step by %g", scale)

            # Predict gradient and trust-region check.
            g_trial = orb_xp.asarray(g_orb, dtype=orb_xp.float64) + orb_xp.asarray(hdxi, dtype=orb_xp.float64)
            norm_g_trial = _norm_f64(g_trial)
            if stat.imic >= 2 and norm_g_trial > norm_gkf * ah_grad_trust_region:
                log.debug("Orbital |g| trust fail; stop micro-iterations.")
                break

            # Commit.
            stat.imic += 1
            dr += dxi
            g_orb = g_trial
            norm_gorb = _norm_f64(g_orb)
            ikf += 1
            x0_guess = dxi.copy()
            log.debug(
                "    orb imic %d  |g|=%3.2e  |dxi|=%3.2e  max=%3.2e",
                stat.imic, norm_gorb, _norm_f64(dxi), max_abs,
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
                norm_gkf_new = _norm_f64(g_kf)

                norm_dg = _norm_f64(g_kf - g_orb)
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
