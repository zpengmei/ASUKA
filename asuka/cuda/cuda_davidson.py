from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import time
import numpy as np


@dataclass
class DavidsonResult:
    converged: np.ndarray  # (nroots,) bool
    e: np.ndarray  # (nroots,) float64
    x: list[np.ndarray]  # list of (n,) float64
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

    def _as_vec_d(x: np.ndarray):
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
        nrm_h = float(np.asarray(cp.asnumpy(nrm), dtype=np.float64))
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

        # For tiny subspace matrices, calling cuSOLVER is often dominated by launch/dispatch overhead.
        # Prefer NumPy eigh on host (matrix is <= ~O(10^2) and transfer is negligible) unless explicitly disabled.
        if subspace_eigh_cpu is None:
            subspace_eigh_cpu = True

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
    if batch_convergence_transfer:
        final_metrics_h = np.asarray(cp.asnumpy(cp.stack((e, conv.astype(cp.float64)), axis=0)), dtype=np.float64)
        e_h = final_metrics_h[0]
        conv_h = np.asarray(final_metrics_h[1] > 0.0, dtype=np.bool_)
    else:
        conv_h = np.asarray(cp.asnumpy(conv), dtype=np.bool_)
        e_h = np.asarray(cp.asnumpy(e), dtype=np.float64)
    x_h_arr = np.asarray(cp.asnumpy(X), dtype=np.float64)
    x_h = [np.asarray(x_h_arr[:, i], dtype=np.float64).copy() for i in range(int(nroots))]
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
    return DavidsonResult(converged=conv_h, e=e_h, x=x_h, niter=int(it), stats=stats)
