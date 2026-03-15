from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import time
import numpy as np


@dataclass
class DavidsonResult:
    converged: Any  # (nroots,) bool
    e: Any  # (nroots,) float64
    x: list[Any]  # list of (n,) float64
    niter: int
    stats: dict[str, Any] | None = None


def davidson_sym_gpu(
    hop,
    *,
    hop_low_precision=None,
    hop_low_precision_threshold: float | None = None,
    hop_low_precision_max_iter: int | None = None,
    x0: list[np.ndarray],
    hdiag: np.ndarray,
    nroots: int = 1,
    max_cycle: int = 50,
    max_space: int = 12,
    tol: float = 1e-10,
    tol_residual: float | None = None,
    lindep: float = 1e-14,
    denom_tol: float = 1e-12,
    stream=None,
    profile: bool = False,
    profile_cuda_sync: bool = False,
    force_final_full_precision_hop: bool = False,
    force_final_full_subspace_refresh: bool = False,
    subspace_eigh_cpu: bool | None = None,
    subspace_eigh_cpu_max_m: int = 64,
    batch_convergence_transfer: bool = True,
    return_cupy: bool = False,
) -> DavidsonResult:
    """Matrix-free symmetric Davidson on GPU (CuPy).

    Parameters
    ----------
    hop
        Callable: `hop(v_d) -> w_d` where both are 1D CuPy arrays (float64).
    x0
        Initial guess vectors on host (NumPy), length >= nroots recommended.
    hdiag
        Diagonal preconditioner (NumPy or CuPy).
    """
    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the GPU Davidson solver") from e

    if stream is None:
        stream = cp.cuda.get_current_stream()

    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be >= 1")
    max_cycle = int(max_cycle)
    max_space = int(max_space)
    tol = float(tol)
    tol_residual = None if tol_residual is None else float(tol_residual)
    lindep = float(lindep)
    denom_tol = float(denom_tol)
    if max_cycle < 1:
        raise ValueError("max_cycle must be >= 1")
    if max_space < nroots:
        raise ValueError("max_space must be >= nroots")
    max_space_eff = int(max_space + (nroots - 1) * 4)
    if max_space_eff < nroots:
        raise ValueError("invalid effective max_space")
    toloose = float(np.sqrt(tol)) if tol_residual is None else float(tol_residual)
    subspace_eigh_cpu_max_m = int(subspace_eigh_cpu_max_m)
    if subspace_eigh_cpu_max_m < 0:
        subspace_eigh_cpu_max_m = 0
    batch_convergence_transfer = bool(batch_convergence_transfer)
    if hop_low_precision_threshold is not None:
        hop_low_precision_threshold = float(hop_low_precision_threshold)
        if hop_low_precision_threshold <= 0.0:
            raise ValueError("hop_low_precision_threshold must be > 0")
        if hop_low_precision is None:
            raise ValueError("hop_low_precision_threshold requires hop_low_precision to be provided")
    if hop_low_precision_max_iter is not None:
        hop_low_precision_max_iter = int(hop_low_precision_max_iter)
        if hop_low_precision_max_iter <= 0:
            raise ValueError("hop_low_precision_max_iter must be > 0 when provided")
        if hop_low_precision is None:
            raise ValueError("hop_low_precision_max_iter requires hop_low_precision to be provided")
    low_precision_active = bool(hop_low_precision is not None)
    low_precision_switch_iter: int | None = None
    low_precision_switch_residual_max: float | None = None
    low_precision_switch_reason: str | None = None

    if not x0:
        raise ValueError("x0 must not be empty")

    stats: dict[str, float] | None = {} if bool(profile) else None
    hop_calls = 0
    hop_time_s = 0.0
    orth_time_s = 0.0
    subspace_time_s = 0.0
    hop_low_precision_calls = 0
    hop_full_precision_calls = 0
    final_full_precision_correction_executed = False
    final_full_precision_correction_rnorm_max: float | None = None
    final_full_subspace_refresh_executed = False
    final_full_subspace_refresh_basis_size: int | None = None

    # GPU event timing for accurate hop measurement
    _gpu_event_pairs: list[tuple] = []  # (start, end) CUDA event pairs for hop calls
    _use_gpu_events = bool(profile) and bool(profile_cuda_sync)

    if isinstance(hdiag, cp.ndarray):
        hdiag_d = cp.asarray(hdiag, dtype=cp.float64).ravel()
    else:
        hdiag_d = cp.asarray(np.asarray(hdiag, dtype=np.float64).ravel(), dtype=cp.float64)
    n = int(hdiag_d.size)
    if n <= 0:
        raise ValueError("invalid hdiag size")

    def _as_vec_d(x):
        if isinstance(x, cp.ndarray):
            v = cp.asarray(x, dtype=cp.float64).ravel()
        else:
            v = cp.asarray(np.asarray(x, dtype=np.float64).ravel(), dtype=cp.float64)
        if v.size != n:
            raise ValueError("x0 vectors must have shape (ncsf,)")
        return v

    def _norm(v):
        return cp.linalg.norm(v)

    def _hop_block(V, *, low_precision: bool):
        """Apply hop to a 1D or 2D CuPy array.

        If `hop` supports 2D inputs (block matvec), use it; otherwise fall back to
        column-wise application. This enables fixed sparse SpMM backends without
        changing the Davidson outer logic.
        """

        nonlocal hop_calls, hop_time_s
        nonlocal hop_low_precision_calls, hop_full_precision_calls
        hop_fn = hop_low_precision if (low_precision and hop_low_precision is not None) else hop
        if V.ndim == 1:
            if stats is not None:
                t_h0 = time.perf_counter()
            if _use_gpu_events:
                _ev_start = cp.cuda.Event(); _ev_end = cp.cuda.Event()
                _ev_start.record(stream)
            W = hop_fn(V)
            _hop_dtype = cp.float32 if low_precision else cp.float64
            W = cp.asarray(W, dtype=_hop_dtype).ravel()
            hop_calls += 1
            if low_precision:
                hop_low_precision_calls += 1
            else:
                hop_full_precision_calls += 1
            if bool(profile_cuda_sync):
                stream.synchronize()
            if _use_gpu_events:
                _ev_end.record(stream)
                _gpu_event_pairs.append((_ev_start, _ev_end))
            if stats is not None:
                hop_time_s += time.perf_counter() - t_h0
            return W

        if V.ndim != 2:
            raise ValueError("hop input must be 1D or 2D")
        m = int(V.shape[1])
        if m <= 0:
            raise ValueError("invalid block size for hop")

        if stats is not None:
            t_h0 = time.perf_counter()
        if _use_gpu_events:
            _ev_start = cp.cuda.Event(); _ev_end = cp.cuda.Event()
            _ev_start.record(stream)
        try:
            W = hop_fn(V)
            if not hasattr(W, "ndim") or int(W.ndim) != 2 or tuple(W.shape) != tuple(V.shape):
                raise ValueError("hop(V) returned an array with unexpected shape for 2D input")
            _hop_dtype = cp.float32 if low_precision else cp.float64
            W = cp.asarray(W, dtype=_hop_dtype)
            hop_calls += m
            if low_precision:
                hop_low_precision_calls += m
            else:
                hop_full_precision_calls += m
            if bool(profile_cuda_sync):
                stream.synchronize()
            if _use_gpu_events:
                _ev_end.record(stream)
                _gpu_event_pairs.append((_ev_start, _ev_end))
            if stats is not None:
                hop_time_s += time.perf_counter() - t_h0
            return W
        except Exception:
            # Fall back to applying hop per column (hop likely only supports 1D inputs).
            _hop_dtype = cp.float32 if low_precision else cp.float64
            cols = [cp.asarray(hop_fn(V[:, i]), dtype=_hop_dtype).ravel() for i in range(m)]
            W = cp.ascontiguousarray(cp.stack(cols, axis=1))
            hop_calls += m
            if low_precision:
                hop_low_precision_calls += m
            else:
                hop_full_precision_calls += m
            if bool(profile_cuda_sync):
                stream.synchronize()
            if _use_gpu_events:
                _ev_end.record(stream)
                _gpu_event_pairs.append((_ev_start, _ev_end))
            if stats is not None:
                hop_time_s += time.perf_counter() - t_h0
            return W

    def _orthonormalize_one(v, *, basis):
        nonlocal orth_time_s
        t0 = None
        if stats is not None:
            t0 = time.perf_counter()
        if basis is not None and int(basis.shape[1]) > 0:
            c = basis.T @ v
            v = v - basis @ c
        nrm = _norm(v)
        nrm_h = float(nrm.item())
        if nrm_h <= lindep:
            if t0 is not None:
                orth_time_s += time.perf_counter() - t0
            return None
        if t0 is not None:
            orth_time_s += time.perf_counter() - t0
        return v / nrm_h

    # Build initial subspace.
    with stream:
        V_buf = cp.empty((n, max_space_eff), dtype=cp.float64)
        W_buf = cp.empty((n, max_space_eff), dtype=cp.float64)
        m = 0
        v_basis = None
        for x in x0:
            v = _as_vec_d(x)
            v = _orthonormalize_one(v, basis=v_basis)
            if v is None:
                continue
            V_buf[:, m] = v
            W_buf[:, m] = cp.asarray(_hop_block(v, low_precision=low_precision_active), dtype=cp.float64).ravel()
            m += 1
            v_basis = V_buf[:, :m]
            if m >= max_space_eff:
                break

        if m < nroots:
            raise RuntimeError("failed to build enough linearly independent initial vectors")

        V = V_buf[:, :m]
        W = W_buf[:, :m]

        conv = cp.zeros((nroots,), dtype=cp.bool_)
        e = cp.zeros((nroots,), dtype=cp.float64)
        e_prev_d = None
        X = None

        if subspace_eigh_cpu is None:
            subspace_eigh_cpu = False

        for it in range(1, max_cycle + 1):
            m = int(V.shape[1])
            if stats is not None:
                t_sub0 = time.perf_counter()
            Hsub = V.T @ W
            Hsub = 0.5 * (Hsub + Hsub.T)
            if bool(subspace_eigh_cpu) and m <= int(subspace_eigh_cpu_max_m):
                Hsub_h = cp.asnumpy(Hsub)
                evals_h, U_h = np.linalg.eigh(Hsub_h)
                e = cp.asarray(evals_h[:nroots], dtype=cp.float64)
                U_r = cp.asarray(U_h[:, :nroots], dtype=cp.float64)
            else:
                evals, U = cp.linalg.eigh(Hsub)
                e = evals[:nroots].copy()
                U_r = U[:, :nroots]
            if bool(profile_cuda_sync):
                stream.synchronize()
            if stats is not None:
                subspace_time_s += time.perf_counter() - t_sub0

            X = V @ U_r
            AX = W @ U_r
            R = AX - X * e[None, :]
            rnorm = cp.linalg.norm(R, axis=0)
            if batch_convergence_transfer:
                if e_prev_d is None:
                    de = cp.zeros_like(e)
                else:
                    de = e - e_prev_d
                conv = (rnorm <= tol) | ((cp.abs(de) <= tol) & (rnorm <= toloose))
                conv_h = np.asarray(cp.asnumpy(conv), dtype=np.bool_)
                e_prev_d = e.copy()
            else:
                e_h = np.asarray(cp.asnumpy(e), dtype=np.float64)
                rnorm_h = np.asarray(cp.asnumpy(rnorm), dtype=np.float64)
                if e_prev_d is None:
                    # Allow convergence on the first iteration based on the residual norm.
                    # Using `de_h=e_h` would make `abs(de_h) <= tol` impossible in the first
                    # iteration and can incorrectly mark an already-converged initial guess
                    # as not converged (then the algorithm may stop with `new_vecs==[]`).
                    de_h = np.zeros_like(e_h)
                else:
                    de_h = e_h - np.asarray(cp.asnumpy(e_prev_d), dtype=np.float64)
                conv_h = (rnorm_h <= tol) | ((np.abs(de_h) <= tol) & (rnorm_h <= toloose))
                e_prev_d = cp.asarray(e_h, dtype=cp.float64)
                conv = cp.asarray(conv_h, dtype=cp.bool_)
            if bool(np.all(conv_h)):
                break

            # 10.16.9: Batch the max rnorm check with convergence transfer to avoid extra sync.
            if low_precision_active and (
                hop_low_precision_threshold is not None or hop_low_precision_max_iter is not None
            ):
                # Use rnorm that was already transferred above in batch_convergence_transfer path,
                # or compute on device and batch with next transfer.
                if batch_convergence_transfer:
                    # rnorm is already on device; get max without extra transfer by using rnorm_h if available
                    rnorm_max_h = float(cp.max(rnorm).item())
                else:
                    rnorm_max_h = float(rnorm_h.max()) if "rnorm_h" in dir() else float(cp.max(rnorm).get())
                switch_by_threshold = bool(
                    hop_low_precision_threshold is not None
                    and rnorm_max_h <= float(hop_low_precision_threshold)
                )
                switch_by_iter = bool(
                    hop_low_precision_max_iter is not None
                    and int(it) >= int(hop_low_precision_max_iter)
                )
                if switch_by_threshold or switch_by_iter:
                    low_precision_active = False
                    if low_precision_switch_iter is None:
                        low_precision_switch_iter = int(it)
                        low_precision_switch_residual_max = float(rnorm_max_h)
                        low_precision_switch_reason = "threshold" if switch_by_threshold else "max_iter"

            new_vecs = []
            for root in range(nroots):
                if bool(conv_h[root]):
                    continue
                r = R[:, root]
                denom = hdiag_d - e[root]
                denom = cp.where(cp.abs(denom) < denom_tol, denom_tol, denom)
                t = r / denom

                t = _orthonormalize_one(t, basis=V)
                if t is None:
                    continue

                if new_vecs:
                    Q = cp.stack(new_vecs, axis=1)
                    t = _orthonormalize_one(t, basis=Q)
                    if t is None:
                        continue

                new_vecs.append(t)

            if not new_vecs:
                break

            if m + len(new_vecs) > max_space_eff:
                # Restart with the current Ritz vectors.
                if X is None:  # pragma: no cover
                    break
                m = int(X.shape[1])
                V_buf[:, :m] = cp.asarray(X, dtype=cp.float64)
                W_buf[:, :m] = cp.asarray(_hop_block(X, low_precision=low_precision_active), dtype=cp.float64)
                V = V_buf[:, :m]
                W = W_buf[:, :m]
                continue

            V_new = cp.stack(new_vecs, axis=1)
            k_new = int(V_new.shape[1])
            W_new = cp.asarray(_hop_block(V_new, low_precision=low_precision_active), dtype=cp.float64)
            V_buf[:, m : m + k_new] = cp.asarray(V_new, dtype=cp.float64)
            W_buf[:, m : m + k_new] = W_new
            m += k_new
            V = V_buf[:, :m]
            W = W_buf[:, :m]

    # Mixed mode correction: evaluate final Ritz vectors with full precision once.
    # Optional full-subspace refresh rebuilds W in full precision before final Ritz extraction.
    if (
        hop_low_precision is not None
        and bool(force_final_full_precision_hop)
        and X is not None
        and int(getattr(X, "shape", (0, 0))[1]) > 0
    ):
        if bool(force_final_full_subspace_refresh) and V is not None and int(getattr(V, "shape", (0, 0))[1]) > 0:
            W_full = cp.ascontiguousarray(_hop_block(V, low_precision=False))
            m_full = int(V.shape[1])
            if stats is not None:
                t_sub0 = time.perf_counter()
            Hsub_full = V.T @ W_full
            Hsub_full = 0.5 * (Hsub_full + Hsub_full.T)
            if bool(subspace_eigh_cpu) and m_full <= int(subspace_eigh_cpu_max_m):
                Hsub_full_h = cp.asnumpy(Hsub_full)
                evals_full_h, U_full_h = np.linalg.eigh(Hsub_full_h)
                e = cp.asarray(evals_full_h[:nroots], dtype=cp.float64)
                U_r_full = cp.asarray(U_full_h[:, :nroots], dtype=cp.float64)
            else:
                evals_full, U_full = cp.linalg.eigh(Hsub_full)
                e = evals_full[:nroots].copy()
                U_r_full = U_full[:, :nroots]
            if bool(profile_cuda_sync):
                stream.synchronize()
            if stats is not None:
                subspace_time_s += time.perf_counter() - t_sub0
            X = V @ U_r_full
            AX_full = W_full @ U_r_full
            final_full_subspace_refresh_executed = True
            final_full_subspace_refresh_basis_size = int(m_full)
        else:
            AX_full = _hop_block(X, low_precision=False)
            e = cp.sum(X * AX_full, axis=0, dtype=cp.float64)
        R_full = AX_full - X * e[None, :]
        rnorm_full = cp.linalg.norm(R_full, axis=0)
        if bool(profile_cuda_sync):
            stream.synchronize()
        final_full_precision_correction_executed = True
        final_full_precision_correction_rnorm_max = float(cp.max(rnorm_full))
        if e_prev_d is None:
            de_full = cp.zeros_like(e)
        else:
            de_full = e - e_prev_d
        conv = (rnorm_full <= tol) | ((cp.abs(de_full) <= tol) & (rnorm_full <= toloose))
        e_prev_d = e.copy()
        low_precision_active = False

    # Export to host.
    if X is None:  # pragma: no cover
        raise RuntimeError("internal error: missing X")
    if bool(return_cupy):
        conv_out = cp.asarray(conv, dtype=cp.bool_)
        e_out = cp.asarray(e, dtype=cp.float64)
        x_out = [cp.ascontiguousarray(X[:, i]) for i in range(int(nroots))]
    else:
        if batch_convergence_transfer:
            final_metrics_h = np.asarray(cp.asnumpy(cp.stack((e, conv.astype(cp.float64)), axis=0)), dtype=np.float64)
            e_out = final_metrics_h[0]
            conv_out = np.asarray(final_metrics_h[1] > 0.0, dtype=np.bool_)
        else:
            conv_out = np.asarray(cp.asnumpy(conv), dtype=np.bool_)
            e_out = np.asarray(cp.asnumpy(e), dtype=np.float64)
        x_h_arr = np.asarray(cp.asnumpy(X), dtype=np.float64)
        x_out = [np.asarray(x_h_arr[:, i], dtype=np.float64).copy() for i in range(int(nroots))]
    if stats is not None:
        stats["hop_calls"] = float(hop_calls)
        stats["hop_time_s"] = float(hop_time_s)
        stats["orth_time_s"] = float(orth_time_s)
        stats["subspace_time_s"] = float(subspace_time_s)
        stats["hop_low_precision_calls"] = float(hop_low_precision_calls)
        stats["hop_full_precision_calls"] = float(hop_full_precision_calls)
        # GPU event timing: aggregate all hop CUDA event pairs
        if _use_gpu_events and _gpu_event_pairs:
            stream.synchronize()
            gpu_hop_ms = 0.0
            for ev_s, ev_e in _gpu_event_pairs:
                ev_e.synchronize()
                gpu_hop_ms += cp.cuda.get_elapsed_time(ev_s, ev_e)
            stats["gpu_hop_time_ms"] = float(gpu_hop_ms)
            stats["gpu_hop_time_s"] = float(gpu_hop_ms / 1000.0)
        stats["hop_low_precision_switch_iter"] = float(-1 if low_precision_switch_iter is None else low_precision_switch_iter)
        stats["hop_low_precision_switch_residual_max"] = float(
            -1.0 if low_precision_switch_residual_max is None else low_precision_switch_residual_max
        )
        stats["hop_low_precision_switch_reason"] = (
            "none" if low_precision_switch_reason is None else str(low_precision_switch_reason)
        )
        stats["hop_low_precision_final_mode"] = "low_precision" if bool(low_precision_active) else "full_precision"
        stats["hop_force_final_full_precision_hop"] = bool(force_final_full_precision_hop)
        stats["hop_force_final_full_subspace_refresh"] = bool(force_final_full_subspace_refresh)
        stats["hop_final_full_subspace_refresh_executed"] = bool(final_full_subspace_refresh_executed)
        stats["hop_final_full_subspace_refresh_basis_size"] = float(
            -1 if final_full_subspace_refresh_basis_size is None else int(final_full_subspace_refresh_basis_size)
        )
        stats["hop_final_full_precision_correction_executed"] = bool(final_full_precision_correction_executed)
        stats["hop_final_full_precision_correction_rnorm_max"] = float(
            -1.0
            if final_full_precision_correction_rnorm_max is None
            else final_full_precision_correction_rnorm_max
        )
    return DavidsonResult(converged=conv_out, e=e_out, x=x_out, niter=int(it), stats=stats)


def jacobi_davidson_sym_gpu(
    hop,
    *,
    x0: list[np.ndarray],
    hdiag: np.ndarray,
    precond=None,
    nroots: int = 1,
    max_cycle: int = 50,
    max_space: int = 12,
    tol: float = 1e-10,
    tol_residual: float | None = None,
    lindep: float = 1e-14,
    denom_tol: float = 1e-12,
    stream=None,
    profile: bool = False,
    profile_cuda_sync: bool = False,
    subspace_eigh_cpu: bool | None = None,
    subspace_eigh_cpu_max_m: int = 64,
    batch_convergence_transfer: bool = True,
    return_cupy: bool = False,
    jd_inner_max_cycle: int = 1,
    jd_inner_tol_rel: float = 0.25,
    jd_keep_corrections: int = 4,
) -> DavidsonResult:
    """Matrix-free symmetric Jacobi-Davidson on GPU (CuPy).

    This keeps the same outer projected-eigensolver contract as
    ``davidson_sym_gpu`` but replaces the diagonal Davidson correction
    with an inexact correction-equation solve for each active root.
    """
    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for the GPU Jacobi-Davidson solver") from e

    if stream is None:
        stream = cp.cuda.get_current_stream()

    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be >= 1")
    max_cycle = int(max_cycle)
    max_space = int(max_space)
    tol = float(tol)
    tol_residual = None if tol_residual is None else float(tol_residual)
    lindep = float(lindep)
    denom_tol = float(denom_tol)
    jd_inner_max_cycle = max(0, int(jd_inner_max_cycle))
    jd_inner_tol_rel = float(jd_inner_tol_rel)
    jd_keep_corrections = max(0, int(jd_keep_corrections))
    if max_cycle < 1:
        raise ValueError("max_cycle must be >= 1")
    if max_space < nroots:
        raise ValueError("max_space must be >= nroots")
    max_space_eff = int(max_space + (nroots - 1) * 4)
    if max_space_eff < nroots:
        raise ValueError("invalid effective max_space")
    toloose = float(np.sqrt(tol)) if tol_residual is None else float(tol_residual)
    subspace_eigh_cpu_max_m = int(subspace_eigh_cpu_max_m)
    if subspace_eigh_cpu_max_m < 0:
        subspace_eigh_cpu_max_m = 0
    batch_convergence_transfer = bool(batch_convergence_transfer)

    if not x0:
        raise ValueError("x0 must not be empty")

    stats: dict[str, float | str | bool] | None = {} if bool(profile) else None
    hop_calls = 0
    hop_time_s = 0.0
    orth_time_s = 0.0
    subspace_time_s = 0.0
    inner_iterations_total = 0
    inner_iterations_max = 0
    restart_count = 0
    locked_count_max = 0
    basis_size_max = 0
    _gpu_event_pairs: list[tuple] = []
    _use_gpu_events = bool(profile) and bool(profile_cuda_sync)
    precond_label = str(getattr(precond, "label", "scalar_diagonal")) if precond is not None else "scalar_diagonal"

    if isinstance(hdiag, cp.ndarray):
        hdiag_d = cp.asarray(hdiag, dtype=cp.float64).ravel()
    else:
        hdiag_d = cp.asarray(np.asarray(hdiag, dtype=np.float64).ravel(), dtype=cp.float64)
    n = int(hdiag_d.size)
    if n <= 0:
        raise ValueError("invalid hdiag size")

    def _as_vec_d(x):
        if isinstance(x, cp.ndarray):
            v = cp.asarray(x, dtype=cp.float64).ravel()
        else:
            v = cp.asarray(np.asarray(x, dtype=np.float64).ravel(), dtype=cp.float64)
        if int(v.size) != int(n):
            raise ValueError("x0 vectors must have shape (ncsf,)")
        return v

    def _hop_block(V):
        nonlocal hop_calls, hop_time_s
        if int(getattr(V, "ndim", 1)) == 1:
            if stats is not None:
                t_h0 = time.perf_counter()
            if _use_gpu_events:
                ev_s = cp.cuda.Event()
                ev_e = cp.cuda.Event()
                ev_s.record(stream)
            W = cp.asarray(hop(V), dtype=cp.float64).ravel()
            hop_calls += 1
            if bool(profile_cuda_sync):
                stream.synchronize()
            if _use_gpu_events:
                ev_e.record(stream)
                _gpu_event_pairs.append((ev_s, ev_e))
            if stats is not None:
                hop_time_s += time.perf_counter() - t_h0
            return W
        if int(V.ndim) != 2:
            raise ValueError("hop input must be 1D or 2D")
        m = int(V.shape[1])
        if m <= 0:
            raise ValueError("invalid block size for hop")
        if stats is not None:
            t_h0 = time.perf_counter()
        if _use_gpu_events:
            ev_s = cp.cuda.Event()
            ev_e = cp.cuda.Event()
            ev_s.record(stream)
        try:
            W = hop(V)
            if not hasattr(W, "ndim") or int(W.ndim) != 2 or tuple(W.shape) != tuple(V.shape):
                raise ValueError("hop(V) returned an array with unexpected shape for 2D input")
            W = cp.asarray(W, dtype=cp.float64)
            hop_calls += m
        except Exception:
            cols = [cp.asarray(hop(V[:, i]), dtype=cp.float64).ravel() for i in range(m)]
            W = cp.ascontiguousarray(cp.stack(cols, axis=1))
            hop_calls += m
        if bool(profile_cuda_sync):
            stream.synchronize()
        if _use_gpu_events:
            ev_e.record(stream)
            _gpu_event_pairs.append((ev_s, ev_e))
        if stats is not None:
            hop_time_s += time.perf_counter() - t_h0
        return W

    def _project_out(v_d, basis_d):
        if basis_d is None or int(getattr(basis_d, "size", 0)) == 0:
            return cp.asarray(v_d, dtype=cp.float64).ravel()
        vec_d = cp.asarray(v_d, dtype=cp.float64).ravel()
        basis_arr = cp.asarray(basis_d, dtype=cp.float64)
        if int(basis_arr.ndim) != 2 or int(basis_arr.shape[1]) == 0:
            return vec_d
        return cp.ascontiguousarray(vec_d - basis_arr @ (basis_arr.T @ vec_d))

    def _stack_cols(cols: list[Any]):
        if not cols:
            return None
        return cp.ascontiguousarray(cp.stack(cols, axis=1))

    def _orthonormalize_one(v_d, *, bases: list[Any | None]):
        nonlocal orth_time_s
        t0 = time.perf_counter() if stats is not None else None
        v = cp.asarray(v_d, dtype=cp.float64).ravel()
        for basis in bases:
            if basis is None:
                continue
            basis_arr = cp.asarray(basis, dtype=cp.float64)
            if int(getattr(basis_arr, "ndim", 1)) != 2 or int(basis_arr.shape[1]) == 0:
                continue
            v = v - basis_arr @ (basis_arr.T @ v)
        nrm_h = float(cp.linalg.norm(v).item())
        if t0 is not None:
            orth_time_s += time.perf_counter() - t0
        if nrm_h <= lindep:
            return None
        return cp.ascontiguousarray(v / nrm_h)

    def _scalar_precond(theta, rhs_d):
        denom = hdiag_d - float(theta)
        sign = cp.where(denom >= 0.0, 1.0, -1.0)
        denom = cp.where(cp.abs(denom) < denom_tol, sign * denom_tol, denom)
        return cp.ascontiguousarray(cp.asarray(rhs_d, dtype=cp.float64).ravel() / denom)

    def _apply_precond(theta, rhs_d, *, root_idx: int):
        if precond is None:
            return _scalar_precond(theta, rhs_d)
        if hasattr(precond, "apply"):
            return cp.ascontiguousarray(precond.apply(theta, rhs_d))
        try:
            return cp.ascontiguousarray(precond(theta, rhs_d, root_idx=int(root_idx)))
        except TypeError:
            return cp.ascontiguousarray(precond(theta, rhs_d))

    def _corr_op(v_d, theta: float, proj_basis_d):
        pv_d = _project_out(v_d, proj_basis_d)
        av_d = cp.asarray(_hop_block(pv_d), dtype=cp.float64).ravel()
        return _project_out(av_d - float(theta) * pv_d, proj_basis_d)

    def _jd_minres_correction(theta: float, rhs_d, *, proj_basis_d, root_idx: int):
        rhs_vec_d = _project_out(rhs_d, proj_basis_d)
        rhs_norm_h = float(cp.linalg.norm(rhs_vec_d).item())
        if rhs_norm_h <= lindep:
            return None, 0, rhs_norm_h
        x_base_d = _project_out(_apply_precond(theta, rhs_vec_d, root_idx=int(root_idx)), proj_basis_d)
        x_base_norm_h = float(cp.linalg.norm(x_base_d).item())
        if x_base_norm_h <= lindep:
            return None, 0, rhs_norm_h
        # Fast inexact JD: use the projected preconditioned residual directly.
        # This keeps the JD outer iteration and block preconditioner but avoids
        # burning extra hop calls on an inner Krylov solve unless explicitly
        # requested via jd_inner_max_cycle > 1.
        if int(jd_inner_max_cycle) <= 1:
            return cp.ascontiguousarray(x_base_d / x_base_norm_h), 0, rhs_norm_h
        r0_d = cp.ascontiguousarray(rhs_vec_d - _corr_op(x_base_d, theta, proj_basis_d))
        beta0_h = float(cp.linalg.norm(r0_d).item())
        inner_tol_h = max(float(tol), min(float(jd_inner_tol_rel), float(np.sqrt(max(rhs_norm_h, 0.0)))) * rhs_norm_h)
        if beta0_h <= inner_tol_h:
            return x_base_d, 0, beta0_h
        v_prev_d = None
        beta_prev_h = 0.0
        v_d = cp.ascontiguousarray(r0_d / beta0_h)
        lanczos_cols: list[Any] = []
        alphas_h: list[float] = []
        betas_h: list[float] = []
        best_y_h: np.ndarray | None = None
        best_m = 0
        best_resid_h = float(beta0_h)
        for inner_it in range(1, int(jd_inner_max_cycle) + 1):
            w_d = _corr_op(v_d, theta, proj_basis_d)
            if v_prev_d is not None:
                w_d = w_d - float(beta_prev_h) * v_prev_d
            alpha_h = float(cp.vdot(v_d, w_d).real.item())
            w_d = w_d - alpha_h * v_d
            prev_cols_d = _stack_cols(lanczos_cols)
            if prev_cols_d is not None:
                w_d = w_d - prev_cols_d @ (prev_cols_d.T @ w_d)
            beta_next_h = float(cp.linalg.norm(w_d).item())
            lanczos_cols.append(cp.ascontiguousarray(v_d))
            alphas_h.append(alpha_h)
            betas_h.append(beta_next_h)
            m_inner = len(lanczos_cols)
            tbar_h = np.zeros((m_inner + 1, m_inner), dtype=np.float64)
            rhs_ls_h = np.zeros((m_inner + 1,), dtype=np.float64)
            rhs_ls_h[0] = float(beta0_h)
            for j in range(m_inner):
                tbar_h[j, j] = float(alphas_h[j])
            for j in range(m_inner - 1):
                tbar_h[j, j + 1] = float(betas_h[j])
                tbar_h[j + 1, j] = float(betas_h[j])
            tbar_h[m_inner, m_inner - 1] = float(betas_h[m_inner - 1])
            y_h, *_ = np.linalg.lstsq(tbar_h, rhs_ls_h, rcond=None)
            resid_h = float(np.linalg.norm(rhs_ls_h - tbar_h @ y_h))
            if resid_h < best_resid_h or best_y_h is None:
                best_y_h = np.asarray(y_h, dtype=np.float64).copy()
                best_m = int(m_inner)
                best_resid_h = float(resid_h)
            if resid_h <= inner_tol_h:
                break
            if beta_next_h <= lindep:
                break
            v_prev_d = cp.ascontiguousarray(v_d)
            beta_prev_h = float(beta_next_h)
            v_d = cp.ascontiguousarray(w_d / beta_next_h)
        if best_y_h is None or best_m <= 0:
            return x_base_d, int(inner_it if "inner_it" in locals() else 0), float(beta0_h)
        v_best_d = _stack_cols(lanczos_cols[:best_m])
        corr_d = x_base_d + v_best_d @ cp.asarray(best_y_h, dtype=cp.float64)
        corr_d = _project_out(corr_d, proj_basis_d)
        return cp.ascontiguousarray(corr_d), int(inner_it if "inner_it" in locals() else 0), float(best_resid_h)

    if subspace_eigh_cpu is None:
        subspace_eigh_cpu = False

    with stream:
        V_buf = cp.empty((n, max_space_eff), dtype=cp.float64)
        W_buf = cp.empty((n, max_space_eff), dtype=cp.float64)
        m = 0
        q_lock_d = None
        lock_e_h: list[float] = []
        v_basis_d = None
        for x in x0:
            v_d = _as_vec_d(x)
            v_d = _orthonormalize_one(v_d, bases=[q_lock_d, v_basis_d])
            if v_d is None:
                continue
            V_buf[:, m] = v_d
            m += 1
            v_basis_d = V_buf[:, :m]
            if m >= max_space_eff:
                break
        if m < nroots:
            raise RuntimeError("failed to build enough linearly independent initial vectors")
        W_buf[:, :m] = cp.asarray(_hop_block(V_buf[:, :m]), dtype=cp.float64)
        V = V_buf[:, :m]
        W = W_buf[:, :m]
        basis_size_max = max(basis_size_max, int(m))
        e_prev_h: np.ndarray | None = None
        it = 0
        last_active_e_h = np.zeros((0,), dtype=np.float64)
        last_active_x_d = None
        last_active_conv_h = np.zeros((0,), dtype=np.bool_)

        for it in range(1, max_cycle + 1):
            m = int(V.shape[1])
            if m <= 0:
                break
            active_need = max(0, int(nroots) - int(len(lock_e_h)))
            if active_need <= 0:
                break
            if stats is not None:
                t_sub0 = time.perf_counter()
            Hsub = V.T @ W
            Hsub = 0.5 * (Hsub + Hsub.T)
            if bool(subspace_eigh_cpu) and m <= int(subspace_eigh_cpu_max_m):
                Hsub_h = cp.asnumpy(Hsub)
                evals_h, U_h = np.linalg.eigh(Hsub_h)
                evals_keep_h = np.asarray(evals_h[:active_need], dtype=np.float64)
                e_act_d = cp.asarray(evals_keep_h, dtype=cp.float64)
                U_r = cp.asarray(U_h[:, :active_need], dtype=cp.float64)
            else:
                evals_d, U_d = cp.linalg.eigh(Hsub)
                e_act_d = cp.asarray(evals_d[:active_need], dtype=cp.float64).ravel()
                U_r = cp.asarray(U_d[:, :active_need], dtype=cp.float64)
                evals_keep_h = np.asarray(cp.asnumpy(e_act_d), dtype=np.float64)
            if bool(profile_cuda_sync):
                stream.synchronize()
            if stats is not None:
                subspace_time_s += time.perf_counter() - t_sub0

            X_d = cp.ascontiguousarray(V @ U_r)
            AX_d = cp.ascontiguousarray(W @ U_r)
            R_d = cp.ascontiguousarray(AX_d - X_d * e_act_d[None, :])
            rnorm_h = np.asarray(cp.asnumpy(cp.linalg.norm(R_d, axis=0)), dtype=np.float64)
            if e_prev_h is None or int(e_prev_h.size) != int(evals_keep_h.size):
                de_h = np.full_like(evals_keep_h, np.inf)
            else:
                de_h = np.asarray(evals_keep_h - e_prev_h, dtype=np.float64)
            conv_h = np.asarray((rnorm_h <= tol) | ((np.abs(de_h) <= tol) & (rnorm_h <= toloose)), dtype=np.bool_)
            if batch_convergence_transfer:
                e_prev_h = np.asarray(evals_keep_h, dtype=np.float64)
            else:
                e_prev_h = np.asarray(evals_keep_h, dtype=np.float64)

            new_locked_cols: list[Any] = []
            new_locked_e_h: list[float] = []
            active_mask = np.ones((int(evals_keep_h.size),), dtype=np.bool_)
            for root in range(int(evals_keep_h.size)):
                if not bool(conv_h[root]):
                    continue
                lock_basis_d = _stack_cols(new_locked_cols)
                x_lock_d = _orthonormalize_one(X_d[:, root], bases=[q_lock_d, lock_basis_d])
                active_mask[root] = False
                if x_lock_d is None:
                    continue
                new_locked_cols.append(x_lock_d)
                new_locked_e_h.append(float(evals_keep_h[root]))
            if new_locked_cols:
                q_new_d = _stack_cols(new_locked_cols)
                if q_lock_d is None or int(getattr(q_lock_d, "size", 0)) == 0:
                    q_lock_d = q_new_d
                else:
                    q_lock_d = cp.ascontiguousarray(cp.concatenate((q_lock_d, q_new_d), axis=1))
                lock_e_h.extend(new_locked_e_h)
                locked_count_max = max(locked_count_max, int(len(lock_e_h)))
            if int(len(lock_e_h)) >= int(nroots):
                last_active_e_h = np.zeros((0,), dtype=np.float64)
                last_active_x_d = None
                last_active_conv_h = np.zeros((0,), dtype=np.bool_)
                break

            e_unconv_h = np.asarray(evals_keep_h[active_mask], dtype=np.float64)
            x_unconv_d = cp.ascontiguousarray(X_d[:, active_mask]) if bool(np.any(active_mask)) else cp.zeros((n, 0), dtype=cp.float64)
            r_unconv_d = cp.ascontiguousarray(R_d[:, active_mask]) if bool(np.any(active_mask)) else cp.zeros((n, 0), dtype=cp.float64)
            last_active_e_h = e_unconv_h
            last_active_x_d = x_unconv_d
            last_active_conv_h = np.zeros((int(e_unconv_h.size),), dtype=np.bool_)
            if int(e_unconv_h.size) == 0:
                break

            proj_basis_d = q_lock_d
            if int(x_unconv_d.shape[1]) > 0:
                proj_basis_d = x_unconv_d if proj_basis_d is None or int(getattr(proj_basis_d, "size", 0)) == 0 else cp.ascontiguousarray(cp.concatenate((proj_basis_d, x_unconv_d), axis=1))

            new_vecs: list[Any] = []
            for root in range(int(e_unconv_h.size)):
                corr_d, inner_it, _inner_resid_h = _jd_minres_correction(
                    float(e_unconv_h[root]),
                    -r_unconv_d[:, root],
                    proj_basis_d=proj_basis_d,
                    root_idx=int(root),
                )
                inner_iterations_total += int(inner_it)
                inner_iterations_max = max(inner_iterations_max, int(inner_it))
                if corr_d is None:
                    continue
                q_new_d = _stack_cols(new_vecs)
                corr_d = _orthonormalize_one(corr_d, bases=[q_lock_d, V, q_new_d])
                if corr_d is None:
                    continue
                new_vecs.append(corr_d)

            if not new_vecs:
                restart_cols = [cp.ascontiguousarray(x_unconv_d[:, i]) for i in range(int(x_unconv_d.shape[1]))]
                v_restart_cols: list[Any] = []
                for col_d in restart_cols:
                    basis_restart_d = _stack_cols(v_restart_cols)
                    v_restart_d = _orthonormalize_one(col_d, bases=[q_lock_d, basis_restart_d])
                    if v_restart_d is not None:
                        v_restart_cols.append(v_restart_d)
                if not v_restart_cols:
                    break
                V_restart_d = cp.ascontiguousarray(cp.stack(v_restart_cols, axis=1))
                W_restart_d = cp.ascontiguousarray(_hop_block(V_restart_d), dtype=cp.float64)
                m_restart = int(V_restart_d.shape[1])
                V_buf[:, :m_restart] = V_restart_d
                W_buf[:, :m_restart] = W_restart_d
                V = V_buf[:, :m_restart]
                W = W_buf[:, :m_restart]
                basis_size_max = max(basis_size_max, int(m_restart))
                restart_count += 1
                continue

            if int(V.shape[1]) + int(len(new_vecs)) > int(max_space_eff):
                restart_cols = [cp.ascontiguousarray(x_unconv_d[:, i]) for i in range(int(x_unconv_d.shape[1]))]
                restart_cols.extend(new_vecs[: int(jd_keep_corrections)])
                v_restart_cols = []
                for col_d in restart_cols:
                    basis_restart_d = _stack_cols(v_restart_cols)
                    v_restart_d = _orthonormalize_one(col_d, bases=[q_lock_d, basis_restart_d])
                    if v_restart_d is not None:
                        v_restart_cols.append(v_restart_d)
                if not v_restart_cols:
                    break
                V_restart_d = cp.ascontiguousarray(cp.stack(v_restart_cols, axis=1))
                W_restart_d = cp.ascontiguousarray(_hop_block(V_restart_d), dtype=cp.float64)
                m_restart = int(V_restart_d.shape[1])
                V_buf[:, :m_restart] = V_restart_d
                W_buf[:, :m_restart] = W_restart_d
                V = V_buf[:, :m_restart]
                W = W_buf[:, :m_restart]
                basis_size_max = max(basis_size_max, int(m_restart))
                restart_count += 1
                continue

            V_new_d = cp.ascontiguousarray(cp.stack(new_vecs, axis=1))
            W_new_d = cp.ascontiguousarray(_hop_block(V_new_d), dtype=cp.float64)
            k_new = int(V_new_d.shape[1])
            m_old = int(V.shape[1])
            V_buf[:, m_old : m_old + k_new] = V_new_d
            W_buf[:, m_old : m_old + k_new] = W_new_d
            V = V_buf[:, : m_old + k_new]
            W = W_buf[:, : m_old + k_new]
            basis_size_max = max(basis_size_max, int(m_old + k_new))

        combined_e_h = np.asarray(lock_e_h + last_active_e_h.tolist(), dtype=np.float64)
        combined_x_cols: list[Any] = []
        if q_lock_d is not None and int(getattr(q_lock_d, "shape", (0, 0))[1]) > 0:
            combined_x_cols.extend([cp.ascontiguousarray(q_lock_d[:, i]) for i in range(int(q_lock_d.shape[1]))])
        if last_active_x_d is not None and int(getattr(last_active_x_d, "shape", (0, 0))[1]) > 0:
            combined_x_cols.extend([cp.ascontiguousarray(last_active_x_d[:, i]) for i in range(int(last_active_x_d.shape[1]))])
        if int(combined_e_h.size) == 0 or not combined_x_cols:
            raise RuntimeError("jacobi_davidson_sym_gpu failed to produce Ritz vectors")
        order_h = np.argsort(combined_e_h, kind="stable")[: int(nroots)]
        combined_e_h = np.asarray(combined_e_h[order_h], dtype=np.float64)
        combined_x_cols = [combined_x_cols[int(i)] for i in order_h.tolist()]
        conv_h = np.asarray([True] * min(len(lock_e_h), int(nroots)) + [False] * max(0, int(nroots) - min(len(lock_e_h), int(nroots))), dtype=np.bool_)[: int(nroots)]

    if bool(return_cupy):
        conv_out = cp.asarray(conv_h, dtype=cp.bool_)
        e_out = cp.asarray(combined_e_h, dtype=cp.float64)
        x_out = [cp.ascontiguousarray(col_d) for col_d in combined_x_cols]
    else:
        conv_out = np.asarray(conv_h, dtype=np.bool_)
        e_out = np.asarray(combined_e_h, dtype=np.float64)
        x_out = [np.asarray(cp.asnumpy(col_d), dtype=np.float64).copy() for col_d in combined_x_cols]
    if stats is not None:
        stats["hop_calls"] = float(hop_calls)
        stats["hop_time_s"] = float(hop_time_s)
        stats["orth_time_s"] = float(orth_time_s)
        stats["subspace_time_s"] = float(subspace_time_s)
        stats["jd_inner_iterations_total"] = float(inner_iterations_total)
        stats["jd_inner_iterations_max"] = float(inner_iterations_max)
        stats["jd_restart_count"] = float(restart_count)
        stats["jd_locked_roots_max"] = float(locked_count_max)
        stats["jd_basis_size_max"] = float(basis_size_max)
        stats["jd_preconditioner"] = str(precond_label)
        if _use_gpu_events and _gpu_event_pairs:
            stream.synchronize()
            gpu_hop_ms = 0.0
            for ev_s, ev_e in _gpu_event_pairs:
                ev_e.synchronize()
                gpu_hop_ms += cp.cuda.get_elapsed_time(ev_s, ev_e)
            stats["gpu_hop_time_ms"] = float(gpu_hop_ms)
            stats["gpu_hop_time_s"] = float(gpu_hop_ms / 1000.0)
    return DavidsonResult(converged=conv_out, e=e_out, x=x_out, niter=int(it), stats=stats)
