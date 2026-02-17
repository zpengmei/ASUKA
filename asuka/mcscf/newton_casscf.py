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
    lindep: float,
    log: _SimpleLogger,
) -> tuple[np.ndarray, float, np.ndarray, int, np.ndarray]:
    w, v, seig = _safe_eigh(heff, ovlp, lindep)
    if w.size == 0 or v.shape[1] == 0:
        return np.zeros_like(xs[0]), 0.0, np.zeros((0,), dtype=np.float64), 0, seig

    idx = np.where(np.abs(v[0]) > 0.1)[0]
    sel = int(idx[0]) if idx.size else int(np.argmax(np.abs(v[0])))
    log.debug1("CIAH eigen-sel %d", sel)
    w_t = float(w[sel])

    v0 = float(v[0, sel])
    if abs(v0) < 1e-14:
        return np.zeros_like(xs[0]), w_t, v[:, sel], sel, seig
    xtrial = _dgemv(v[1:, sel] / v0, xs)
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

    w_t = 0.0
    for istep in range(max_cycle):
        g = np.asarray(g_op(), dtype=np.float64).ravel()
        nx = len(xs_l)
        for i in range(nx):
            heff[i + 1, 0] = float(dot(xs_l[i].conj(), g).real)
            heff[nx, i + 1] = float(dot(xs_l[nx - 1].conj(), ax_l[i]).real)
            ovlp[nx, i + 1] = float(dot(xs_l[nx - 1].conj(), xs_l[i]).real)
        heff[0, : nx + 1] = heff[: nx + 1, 0].conj()
        heff[1:nx, nx] = heff[nx, 1:nx].conj()
        ovlp[1:nx, nx] = ovlp[nx, 1:nx].conj()

        nvec = nx + 1
        wlast = w_t
        xtrial, w_t, v_t, index, seig = _regular_step(heff[:nvec, :nvec], ovlp[:nvec, :nvec], xs_l, lindep, log)
        s0 = float(seig[0]) if seig.size else 0.0
        if v_t.size == 0:
            z = np.zeros_like(x0)
            yield True, istep + 1, w_t, z, z, z, s0
            break

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
    h1e_mo = _to_np_f64((mo.T @ hcore) @ mo)

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
    """cuGUGA-owned `gen_g_hop` implementation (no PySCF newton proxy).

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

    def g_update(u: np.ndarray, fcivec: Any) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        if u.ndim != 2 or u.shape[0] != cache.nmo or u.shape[1] != cache.nmo:
            raise ValueError("u must be (nmo,nmo)")
        xp_mo, _ = _get_xp(mo)
        mo1 = _to_xp_f64(mo, xp_mo) @ xp_mo.asarray(u, dtype=xp_mo.float64)
        eris1 = casscf.ao2mo(mo1)
        g1, _gup, _hop, _diag = gen_g_hop_internal(
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
        return np.asarray(g1, dtype=np.float64).ravel()

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

    # CPU copies (always needed for fallback / CPU-only callers).
    ppaa_cpu = _to_np_f64(_ppaa_hop)
    papa_cpu = _to_np_f64(_papa_hop)

    # GPU copies of tensors used inside _h_op_raw (one-time upload).
    if _hop_on_gpu:
        ppaa_dev = _to_xp_f64(_ppaa_hop, _hop_xp)
        papa_dev = _to_xp_f64(_papa_hop, _hop_xp)
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

    def _h_op_raw(x):
        xp, on_gpu = _get_xp(x)
        x = xp.asarray(x, dtype=xp.float64).ravel()

        # Select device-appropriate tensors.
        if on_gpu:
            ppaa, papa, paaa = ppaa_dev, papa_dev, paaa_dev
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
            ppaa, papa, paaa = ppaa_cpu, papa_cpu, cache.paaa
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
        if cache.ncore:
            vhf_a = (
                xp.einsum("pquv,uv->pq", ppaa[:, : cache.ncore], tdm1, optimize=True)
                - 0.5 * xp.einsum("puqv,uv->pq", papa[:, :, : cache.ncore], tdm1, optimize=True)
            )
        else:
            vhf_a = xp.empty((cache.nmo, 0), dtype=xp.float64)

        jk = (
            xp.einsum("pquv,pq->uv", ppaa, ddm_c, optimize=True)
            - 0.5 * xp.einsum("puqv,pq->uv", papa, ddm_c, optimize=True)
        )

        g_dm2 = xp.einsum("puwx,wxuv->pv", paaa, tdm2, optimize=True)

        aaaa = (ra.T @ paaa.reshape(cache.nmo, -1)).reshape((cache.ncas,) * 4)
        aaaa = aaaa + aaaa.transpose(1, 0, 2, 3)
        aaaa = aaaa + aaaa.transpose(2, 3, 0, 1)

        # ── AO-basis DF-JK (GPU stays on GPU, CPU launches async) ──
        if on_gpu:
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
        return xp.asarray(out, dtype=xp.float64).ravel()

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

    def precond(x: np.ndarray, e: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if callable(h_diag):
            x = np.asarray(h_diag(x, e - casscf.ah_level_shift), dtype=np.float64).ravel()
        else:
            hdiagd = np.asarray(h_diag, dtype=np.float64).ravel() - (e - casscf.ah_level_shift)
            hdiagd[np.abs(hdiagd) < 1e-8] = 1e-8
            x = x / hdiagd
        nrm = float(np.linalg.norm(x))
        if nrm > 0.0:
            x *= 1.0 / nrm
        return x

    def scale_down_step(dxi: np.ndarray, hdxi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dxmax = float(np.max(np.abs(dxi)))
        if dxmax > float(max_stepsize):
            scale = float(max_stepsize) / dxmax
            log.debug1("Scale rotation by %g", scale)
            dxi = np.asarray(dxi, dtype=np.float64) * scale
            hdxi = np.asarray(hdxi, dtype=np.float64) * scale
        return dxi, hdxi

    stat = NewtonMicroStats(imic=0, tot_hop=0, tot_kf=1)
    dr = np.zeros_like(g_all)
    ikf = 0
    u = np.eye(nmo, dtype=np.float64)
    ci_kf: Any = ci0_use

    if x0_guess is None:
        x0_guess = g_all
    x0_guess = np.asarray(x0_guess, dtype=np.float64).ravel()
    g_op = lambda: g_all

    if norm_gall < float(conv_tol_grad) * 0.3:
        return u, ci_kf, norm_gall, stat, x0_guess

    last_dxi = x0_guess
    for ah_conv, ihop, w, dxi, hdxi, residual, seig in davidson_cc(
        h_op,
        g_op,
        precond,
        x0_guess,
        tol=casscf.ah_conv_tol,
        max_cycle=casscf.ah_max_cycle,
        lindep=casscf.ah_lindep,
        verbose=log,
    ):
        stat.tot_hop = int(ihop)
        norm_residual = float(np.linalg.norm(residual))
        if (
            bool(ah_conv)
            or int(ihop) == int(casscf.ah_max_cycle)
            or ((norm_residual < float(casscf.ah_start_tol)) and (int(ihop) >= int(casscf.ah_start_cycle)))
            or (float(seig) < float(casscf.ah_lindep))
        ):
            stat.imic += 1
            last_dxi = np.asarray(dxi, dtype=np.float64).ravel()
            dxi_s, hdxi_s = scale_down_step(last_dxi, np.asarray(hdxi, dtype=np.float64).ravel())

            dr += dxi_s
            g_all = g_all + hdxi_s
            norm_dr = float(np.linalg.norm(dr))
            norm_gall = float(np.linalg.norm(g_all))
            norm_gorb = float(np.linalg.norm(g_all[:ngorb]))
            norm_gci = float(np.linalg.norm(g_all[ngorb:]))
            log.debug(
                "    imic %d(%d)  |g|=%3.2e (%2.1e %2.1e)  |dxi|=%3.2e max(x)=%3.2e |dr|=%3.2e  eig=%2.1e seig=%2.1e",
                stat.imic,
                ihop,
                norm_gall,
                norm_gorb,
                norm_gci,
                float(np.linalg.norm(dxi_s)),
                float(np.max(np.abs(dxi_s))),
                norm_dr,
                float(w),
                float(seig),
            )

            max_cycle = max(
                int(casscf.max_cycle_micro),
                int(casscf.max_cycle_micro) - int(np.log(norm_gkf + 1e-7) * 2),
            )
            max_cycle_cap = getattr(casscf, "ah_max_cycle_micro_cap", None)
            if max_cycle_cap is not None:
                max_cycle = min(max_cycle, max(int(casscf.max_cycle_micro), int(max_cycle_cap)))
            log.debug1("Set max_cycle %d", max_cycle)
            ikf += 1

            if stat.imic > 3 and norm_gall > norm_gkf * float(casscf.ah_grad_trust_region):
                g_all = g_all - hdxi_s
                dr -= dxi_s
                norm_gall = float(np.linalg.norm(g_all))
                log.debug("|g| >> keyframe, Restore previous step")
                break

            if stat.imic >= max_cycle or norm_gall < float(conv_tol_grad) * 0.3:
                break

            if ikf >= max(
                int(casscf.kf_interval),
                int(casscf.kf_interval) - int(np.log(norm_dr + 1e-7)),
            ) or (norm_gall < norm_gkf / float(casscf.kf_trust_region)):
                ikf = 0
                u, ci_kf = extract_rotation(casscf, dr, u, ci_kf, ci_update=ci_update)
                dr[:] = 0.0
                g_kf1 = np.asarray(g_update(u, ci_kf), dtype=np.float64).ravel()
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

                if (
                    norm_dg < norm_gall * float(casscf.ah_grad_trust_region)
                    or norm_gkf1 < float(conv_tol_grad) * float(casscf.ah_grad_trust_region)
                ):
                    g_all = g_kf1
                    norm_gall = norm_gkf = norm_gkf1
                else:
                    g_all = g_all - hdxi_s
                    dr -= dxi_s
                    norm_gall = norm_gkf = float(np.linalg.norm(g_all))
                    log.debug("Out of trust region. Restore previous step")
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


__all__ = [
    "WeightsInfo",
    "NewtonMicroStats",
    "compute_ci_gram_inv",
    "extract_rotation",
    "gen_g_hop_internal",
    "gen_g_hop",
    "kernel_newton",
    "kernel_newton_inplace",
    "update_orb_ci",
    "pack_ci_list",
    "project_ci_root_span",
    "unpack_ci_list",
]
