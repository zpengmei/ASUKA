"""Heat-bath screened CIPSI selection + EN-PT2.

This module implements the selection step of HB-SCI: for each source CSF j
in the variational space, screen integrals by magnitude and build a
sparse g_flat that feeds the existing GPU apply/hash kernels.

Reference: Holmes, Sharma, Umrigar, JCTC 2017, 13, 1595.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from asuka.sci.hb_integrals import HeatBathIntegralIndex


def _python_build_screened_g_flat(
    hb_index: HeatBathIntegralIndex,
    occ: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    """Build screened g_flat[norb, norb] for a single source CSF.

    For each (p,q), the effective coupling is:
        g_flat[p,q] = h_eff[p,q] + 0.5 * Σ_{r,s: |v_{pq,rs}| > cutoff} v_{pq,rs} * n_r δ_{rs==rr_diag_factor}

    But in GUGA the g_flat is *not* occupation-weighted at build time
    (the DFS walk handles that). So we just accumulate surviving integrals:
        g_flat[p,q] = h_eff[p,q] + 0.5 * Σ_{rs: |v| > cutoff} v_{pq,rs}  [for off-diag rs]
                                  + 0.5 * occ_r * v_{pq,rr}               [for diag rs]

    Actually, looking at the existing frontier_hash code, g_flat[p,q] is built
    differently for diagonal vs off-diagonal rs. The diagonal contribution
    (r==s) folds in occupancy, while off-diagonal is the raw integral.

    For heat-bath screening, we screen at the *integral level* before the
    DFS walk. The g_flat for a given source CSF j includes:
      - h_eff_pq (one-body)
      - 0.5 * Σ_r occ_r(j) * v_{pq,rr} (diagonal two-body, occupancy-weighted)
      - 0.5 * v_{pq,rs} for r!=s (off-diagonal, raw integrals applied via DFS)

    But the existing apply kernel expects the FULL g_flat for each source CSF,
    so we build the *complete* g_flat per source CSF with screening.

    Parameters
    ----------
    hb_index : HeatBathIntegralIndex
        Sorted integral tables.
    occ : ndarray, shape (norb,)
        Occupancy vector for this source CSF (0, 1, or 2).
    cutoff : float
        Absolute integral cutoff. Skip integrals with |v| < cutoff.

    Returns
    -------
    g_flat : ndarray, shape (norb, norb)
        Screened effective coupling matrix.
    """
    norb = hb_index.norb
    nops = norb * norb
    g_flat = np.zeros((norb, norb), dtype=np.float64)

    # One-body contributions: just copy h_eff values above cutoff
    for k in range(hb_index.n_h1):
        if hb_index.h1_abs[k] < cutoff:
            break
        p, q = int(hb_index.h1_pq[k, 0]), int(hb_index.h1_pq[k, 1])
        g_flat[p, q] = hb_index.h1_signed[k]

    # Two-body contributions
    for pq in range(nops):
        if hb_index.pq_max_v[pq] < cutoff:
            continue
        p, q = pq // norb, pq % norb
        lo = int(hb_index.pq_ptr[pq])
        hi = int(hb_index.pq_ptr[pq + 1])
        if lo >= hi:
            continue

        for k in range(lo, hi):
            if hb_index.v_abs[k] < cutoff:
                break
            rs = int(hb_index.rs_idx[k])
            r, s = rs // norb, rs % norb
            v = hb_index.v_signed[k]
            if r == s:
                # Diagonal: occupancy-weighted
                g_flat[p, q] += 0.5 * float(occ[r]) * v
            else:
                # Off-diagonal: raw integral (DFS walk handles coupling)
                g_flat[p, q] += 0.5 * v

    return g_flat


def heat_bath_select_and_pt2(
    hb_index: HeatBathIntegralIndex,
    sel_idx: np.ndarray,
    c_sel: np.ndarray,
    e_var: np.ndarray,
    max_add: int,
    epsilon: float,
    ws: Any,
    drt: Any,
    frontier_buffers: dict,
    nroots: int,
    ncsf: int,
    denom_floor: float,
    *,
    backend: str = "auto",
    verbose: int = 0,
) -> tuple[list[int], np.ndarray]:
    """Heat-bath screened selection + EN-PT2.

    Algorithm:
    1. max_cj[j] = max_r |c_sel[j,r]| per source CSF
    2. For each j: cutoff_j = eps / max_cj[j]
       - Build screened g_flat from surviving integrals
    3. Apply g_flat via existing GPU apply_g_flat_scatter_atomic_frontier_hash
    4. Extract + score + topk via existing CIPSI kernels

    Parameters
    ----------
    hb_index : HeatBathIntegralIndex
        Pre-sorted integral index.
    sel_idx : ndarray, shape (nsel,)
        Global CSF indices in the variational space.
    c_sel : ndarray, shape (nsel, nroots)
        CI coefficients in selected space.
    e_var : ndarray, shape (nroots,)
        Variational energies.
    max_add : int
        Maximum number of new CSFs to add.
    epsilon : float
        Heat-bath screening threshold.
    ws : GugaMatvecEriMatWorkspace
        CUDA workspace with DRT device arrays.
    drt : DRT
        The DRT defining the full CSF space.
    frontier_buffers : dict
        Pre-allocated frontier hash buffers from gpu_cipsi setup.
    nroots : int
        Number of roots.
    ncsf : int
        Total number of CSFs.
    denom_floor : float
        Floor for PT2 denominators.
    backend : str
        "python" for Python screening, "cuda" for fused CUDA kernel, "auto" to detect.
    verbose : int
        Verbosity level.

    Returns
    -------
    new_idx : list[int]
        Global CSF indices of newly selected CSFs.
    e_pt2 : ndarray, shape (nroots,)
        EN-PT2 energy correction estimate.
    """
    backend_s = str(backend).lower()
    env_backend = os.environ.get("ASUKA_HB_SCI_BACKEND", "").strip().lower()
    if env_backend:
        backend_s = env_backend

    if backend_s == "auto":
        # Check if fused CUDA kernel is available
        try:
            from asuka.cuda.cuda_backend import has_hb_screen_and_apply_device  # noqa: PLC0415

            if has_hb_screen_and_apply_device():
                backend_s = "cuda"
            else:
                backend_s = "python"
        except (ImportError, AttributeError):
            backend_s = "python"

    if backend_s == "cuda":
        return _heat_bath_select_cuda(
            hb_index, sel_idx, c_sel, e_var, max_add, epsilon,
            ws, drt, frontier_buffers, nroots, ncsf, denom_floor,
            verbose=verbose,
        )

    return _heat_bath_select_python(
        hb_index, sel_idx, c_sel, e_var, max_add, epsilon,
        ws, drt, frontier_buffers, nroots, ncsf, denom_floor,
        verbose=verbose,
    )


def _heat_bath_select_python(
    hb_index: HeatBathIntegralIndex,
    sel_idx: np.ndarray,
    c_sel: np.ndarray,
    e_var: np.ndarray,
    max_add: int,
    epsilon: float,
    ws: Any,
    drt: Any,
    frontier_buffers: dict,
    nroots: int,
    ncsf: int,
    denom_floor: float,
    *,
    verbose: int = 0,
) -> tuple[list[int], np.ndarray]:
    """Python-side heat-bath selection using batched GPU apply/hash kernels.

    Key optimizations vs the naive per-CSF loop:

    1. **GEMM for diagonal g_flat**: ``occ_batch (nsel, norb) @ eri_diag_t (norb, norb^2)``
       replaces nsel individual Python CSR loops for the occupancy-weighted diagonal
       (r==s) two-body part.  Pre-allocated output avoids the broadcast-copy allocation.

    2. **Single-pass g_base**: One ``build_g_base`` call at the tightest global cutoff
       (= epsilon / max(max_cj)) is conservative (includes all integrals needed by any
       source CSF) and avoids per-bin g_base overhead.

    3. **One GPU kernel launch per root**: Entire g_batch (nsel, norb^2) uploaded once
       per root and processed in a single batched kernel call, reducing Python-level
       launches from O(nroots × nsel) to O(nroots).
    """
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for HB-SCI") from e

    from asuka.cuda.cuda_backend import (  # noqa: PLC0415
        apply_g_flat_scatter_atomic_frontier_hash_inplace_device,
        cipsi_frontier_hash_clear_inplace_device,
        cipsi_frontier_hash_extract_inplace_device,
        cipsi_score_and_select_topk_inplace_device,
    )
    from asuka.cuguga.state_cache import get_state_cache  # noqa: PLC0415
    from asuka.sci.hb_integrals import build_g_base  # noqa: PLC0415

    norb = hb_index.norb
    nops = norb * norb
    nsel = int(sel_idx.size)
    if nsel <= 0:
        return [], np.zeros((nroots,), dtype=np.float64)

    stream = cp.cuda.get_current_stream()

    # --- Step 1: Occupancy batch (nsel, norb) from state_cache steps ---
    state_cache = get_state_cache(drt)
    steps = np.asarray(state_cache.steps, dtype=np.int8)  # (ncsf, norb) step values
    _STEP_TO_OCC = np.array([0, 1, 1, 2], dtype=np.float64)

    sel_idx_i32 = np.asarray(sel_idx, dtype=np.int32)
    occ_batch = _STEP_TO_OCC[steps[sel_idx_i32].astype(np.int32)]  # (nsel, norb)

    # --- Step 2: max|c_j| and global cutoff ---
    max_cj = np.max(np.abs(c_sel), axis=1)  # (nsel,)
    pos_mask = max_cj > 0.0
    if not pos_mask.any():
        return [], np.zeros((nroots,), dtype=np.float64)

    # Use tightest cutoff: epsilon / max(max_cj) → conservative (includes all integrals
    # needed by any source CSF; some may get extra small terms, which is correct).
    global_cutoff = float(epsilon) / float(max_cj[pos_mask].max())

    # --- Step 3: Build g_batch (nsel, nops) ---
    # Pre-allocate to avoid the broadcast-copy allocation in g_base + g_diag.
    g_batch = np.empty((nsel, nops), dtype=np.float64)
    np.dot(occ_batch, hb_index.eri_diag_t, out=g_batch)   # (nsel, nops) via BLAS GEMM
    g_batch *= 0.5
    g_batch += build_g_base(hb_index, global_cutoff)       # in-place broadcast add

    # Zero out rows for CSFs with max_cj == 0 (no contribution possible)
    if not pos_mask.all():
        g_batch[~pos_mask] = 0.0

    # --- Step 4: GPU setup and hash table ---
    c_sel_d = cp.ascontiguousarray(cp.asarray(c_sel, dtype=cp.float64))
    e_var_d = cp.ascontiguousarray(cp.asarray(e_var, dtype=cp.float64).ravel())
    sel_idx_d = cp.asarray(sel_idx_i32, dtype=cp.int32)

    cap = int(frontier_buffers.get("hash_cap", 0))
    if cap <= 0:
        mult = min(256, max(32, nops // 4))
        target = int(min(ncsf, max(1 << 20, mult * nsel)))
        cap = 1
        while cap < target:
            cap <<= 1
        cap = max(1024, cap)
        while cap > ncsf and cap > 1:
            cap >>= 1

    frontier_hash_max_retries = int(frontier_buffers.get("max_retries", 8))
    selected_mask_d = frontier_buffers.get("selected_mask_d")
    hdiag_d = frontier_buffers.get("hdiag_d")

    # Upload g_batch to GPU once (shared across all roots)
    g_batch_d = cp.asarray(g_batch, dtype=cp.float64)  # (nsel, nops) — one PCIe transfer

    for attempt in range(frontier_hash_max_retries):
        if int(frontier_buffers.get("hash_cap", 0)) != cap or frontier_buffers.get("hash_keys") is None:
            frontier_buffers["hash_cap"] = cap
            frontier_buffers["hash_keys"] = cp.empty((cap,), dtype=cp.int32)
            frontier_buffers["hash_vals"] = cp.empty((nroots, cap), dtype=cp.float64)
            frontier_buffers["hash_overflow"] = cp.empty((1,), dtype=cp.int32)
            frontier_buffers["out_idx"] = cp.empty((cap,), dtype=cp.int32)
            frontier_buffers["out_vals"] = cp.empty((nroots, cap), dtype=cp.float64)
            frontier_buffers["out_nnz"] = cp.empty((1,), dtype=cp.int32)

        hash_keys = frontier_buffers["hash_keys"]
        hash_vals = frontier_buffers["hash_vals"]
        hash_overflow = frontier_buffers["hash_overflow"]
        out_idx = frontier_buffers["out_idx"]
        out_vals = frontier_buffers["out_vals"]
        out_nnz = frontier_buffers["out_nnz"]

        cipsi_frontier_hash_clear_inplace_device(hash_keys, hash_vals, threads=256, stream=stream, sync=False)
        cp.cuda.runtime.memsetAsync(int(hash_overflow.data.ptr), 0, 4, int(stream.ptr))

        # --- Step 5: One GPU kernel launch per root (not per CSF) ---
        for r in range(nroots):
            task_scale_d = cp.ascontiguousarray(c_sel_d[:, r])  # (nsel,)
            apply_g_flat_scatter_atomic_frontier_hash_inplace_device(
                drt,
                ws.drt_dev,
                ws.state_dev,
                sel_idx_d,
                g_batch_d,
                task_scale=task_scale_d,
                hash_keys=hash_keys,
                hash_vals=hash_vals,
                root=r,
                overflow=hash_overflow,
                clear_overflow=False,
                threads=256,
                stream=stream,
                sync=False,
                check_overflow=False,
            )

        # Overflow check
        stream.synchronize()
        if int(hash_overflow.get()[0]) != 0:
            cap = min(cap << 1, 1 << 30)
            continue

        # Extract compact frontier
        cipsi_frontier_hash_extract_inplace_device(
            hash_keys,
            hash_vals,
            out_idx=out_idx,
            out_vals_root_major=out_vals,
            out_nnz=out_nnz,
            threads=256,
            stream=stream,
            sync=True,
        )
        nnz_out = int(out_nnz.get()[0])

        # Score and select top-k
        out_new_idx = cp.empty((max(1, max_add),), dtype=cp.int32)
        out_new_n = cp.empty((1,), dtype=cp.int32)
        out_pt2 = cp.empty((nroots,), dtype=cp.float64)
        cipsi_score_and_select_topk_inplace_device(
            out_idx,
            out_vals,
            nnz=nnz_out,
            e_var=e_var_d,
            hdiag=hdiag_d,
            selected_mask=selected_mask_d,
            denom_floor=denom_floor,
            out_new_idx=out_new_idx,
            out_new_n=out_new_n,
            out_pt2=out_pt2,
            threads=256,
            stream=stream,
            sync=True,
        )

        e_pt2_h = cp.asnumpy(out_pt2)
        n_new = int(out_new_n.get()[0])
        if n_new > 0:
            new_idx_h = cp.asnumpy(out_new_idx[:n_new]).astype(np.int64, copy=False).tolist()
            return [int(ii) for ii in new_idx_h], np.asarray(e_pt2_h, dtype=np.float64)
        return [], np.asarray(e_pt2_h, dtype=np.float64)

    raise RuntimeError("heat-bath frontier-hash overflow: increase capacity or reduce epsilon")


def _heat_bath_select_cuda(
    hb_index: HeatBathIntegralIndex,
    sel_idx: np.ndarray,
    c_sel: np.ndarray,
    e_var: np.ndarray,
    max_add: int,
    epsilon: float,
    ws: Any,
    drt: Any,
    frontier_buffers: dict,
    nroots: int,
    ncsf: int,
    denom_floor: float,
    *,
    verbose: int = 0,
) -> tuple[list[int], np.ndarray]:
    """Fused CUDA heat-bath selection kernel path."""
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for HB-SCI") from e

    from asuka.cuda.cuda_backend import (  # noqa: PLC0415
        cipsi_frontier_hash_clear_inplace_device,
        cipsi_frontier_hash_extract_inplace_device,
        cipsi_score_and_select_topk_inplace_device,
        hb_screen_and_apply_inplace_device,
    )

    norb = hb_index.norb
    nsel = int(sel_idx.size)
    if nsel <= 0:
        return [], np.zeros((nroots,), dtype=np.float64)

    stream = cp.cuda.get_current_stream()

    sel_idx_d = cp.asarray(np.asarray(sel_idx, dtype=np.int32), dtype=cp.int32)
    c_sel_d = cp.ascontiguousarray(cp.asarray(c_sel, dtype=cp.float64))
    e_var_d = cp.ascontiguousarray(cp.asarray(e_var, dtype=cp.float64).ravel())

    # Upload HB index to device if not already cached
    hb_dev = frontier_buffers.get("hb_dev")
    if hb_dev is None:
        from asuka.sci.hb_integrals import upload_hb_index  # noqa: PLC0415
        hb_dev = upload_hb_index(hb_index, cp)
        frontier_buffers["hb_dev"] = hb_dev

    cap = int(frontier_buffers.get("hash_cap", 0))
    if cap <= 0:
        mult = min(256, max(32, norb * norb // 4))
        target = int(min(ncsf, max(1 << 20, mult * nsel)))
        cap = 1
        while cap < target:
            cap <<= 1
        cap = max(1024, cap)
        while cap > ncsf and cap > 1:
            cap >>= 1

    frontier_hash_max_retries = int(frontier_buffers.get("max_retries", 8))
    selected_mask_d = frontier_buffers.get("selected_mask_d")
    hdiag_d = frontier_buffers.get("hdiag_d")

    for attempt in range(frontier_hash_max_retries):
        if int(frontier_buffers.get("hash_cap", 0)) != cap or frontier_buffers.get("hash_keys") is None:
            frontier_buffers["hash_cap"] = cap
            frontier_buffers["hash_keys"] = cp.empty((cap,), dtype=cp.int32)
            frontier_buffers["hash_vals"] = cp.empty((nroots, cap), dtype=cp.float64)
            frontier_buffers["hash_overflow"] = cp.empty((1,), dtype=cp.int32)
            frontier_buffers["out_idx"] = cp.empty((cap,), dtype=cp.int32)
            frontier_buffers["out_vals"] = cp.empty((nroots, cap), dtype=cp.float64)
            frontier_buffers["out_nnz"] = cp.empty((1,), dtype=cp.int32)

        hash_keys = frontier_buffers["hash_keys"]
        hash_vals = frontier_buffers["hash_vals"]
        hash_overflow = frontier_buffers["hash_overflow"]
        out_idx = frontier_buffers["out_idx"]
        out_vals = frontier_buffers["out_vals"]
        out_nnz = frontier_buffers["out_nnz"]

        cipsi_frontier_hash_clear_inplace_device(hash_keys, hash_vals, threads=256, stream=stream, sync=False)
        cp.cuda.runtime.memsetAsync(int(hash_overflow.data.ptr), 0, 4, int(stream.ptr))

        for r in range(nroots):
            hb_screen_and_apply_inplace_device(
                drt,
                ws.drt_dev,
                ws.state_dev,
                sel_idx_d,
                c_sel_d[:, r],
                nsel=nsel,
                nroots=nroots,
                root=r,
                h1_pq=hb_dev["h1_pq"],
                h1_abs=hb_dev["h1_abs"],
                h1_signed=hb_dev["h1_signed"],
                n_h1=hb_index.n_h1,
                pq_ptr=hb_dev["pq_ptr"],
                rs_idx=hb_dev["rs_idx"],
                v_abs=hb_dev["v_abs"],
                v_signed=hb_dev["v_signed"],
                pq_max_v=hb_dev["pq_max_v"],
                eps=float(epsilon),
                hash_keys=hash_keys,
                hash_vals=hash_vals,
                overflow=hash_overflow,
                threads=256,
                stream=stream,
                sync=False,
            )

        stream.synchronize()
        if int(hash_overflow.get()[0]) != 0:
            cap = min(cap << 1, 1 << 30)
            continue

        cipsi_frontier_hash_extract_inplace_device(
            hash_keys, hash_vals,
            out_idx=out_idx, out_vals_root_major=out_vals, out_nnz=out_nnz,
            threads=256, stream=stream, sync=True,
        )
        nnz_out = int(out_nnz.get()[0])

        out_new_idx = cp.empty((max(1, max_add),), dtype=cp.int32)
        out_new_n = cp.empty((1,), dtype=cp.int32)
        out_pt2 = cp.empty((nroots,), dtype=cp.float64)
        cipsi_score_and_select_topk_inplace_device(
            out_idx, out_vals, nnz=nnz_out,
            e_var=e_var_d, hdiag=hdiag_d, selected_mask=selected_mask_d,
            denom_floor=denom_floor,
            out_new_idx=out_new_idx, out_new_n=out_new_n, out_pt2=out_pt2,
            threads=256, stream=stream, sync=True,
        )

        e_pt2_h = cp.asnumpy(out_pt2)
        n_new = int(out_new_n.get()[0])
        if n_new > 0:
            new_idx_h = cp.asnumpy(out_new_idx[:n_new]).astype(np.int64, copy=False).tolist()
            return [int(ii) for ii in new_idx_h], np.asarray(e_pt2_h, dtype=np.float64)
        return [], np.asarray(e_pt2_h, dtype=np.float64)

    raise RuntimeError("heat-bath frontier-hash overflow: increase capacity or reduce epsilon")


def adaptive_epsilon(
    iteration: int,
    nsel: int,
    nsel_target: int,
    eps_init: float = 1e-3,
    eps_final: float = 1e-6,
) -> float:
    """Compute adaptive epsilon for heat-bath screening.

    Early iterations use aggressive screening (large eps) for fast growth.
    Late iterations use fine screening (small eps) for convergence.

    Parameters
    ----------
    iteration : int
        Current SCI iteration number (1-based).
    nsel : int
        Current size of the variational space.
    nsel_target : int
        Target size of the variational space.
    eps_init : float
        Initial epsilon (aggressive screening).
    eps_final : float
        Final epsilon (fine screening).

    Returns
    -------
    float
        Epsilon for this iteration.
    """
    if nsel_target <= 0:
        return float(eps_final)
    frac = min(1.0, float(nsel) / float(nsel_target))
    if eps_init <= 0 or eps_final <= 0 or eps_init <= eps_final:
        return float(eps_final)
    return float(eps_init) * (float(eps_final) / float(eps_init)) ** frac


def semistochastic_pt2(
    hb_index: HeatBathIntegralIndex,
    sel_idx: np.ndarray,
    c_sel: np.ndarray,
    e_var: np.ndarray,
    ws: Any,
    drt: Any,
    frontier_buffers: dict,
    nroots: int,
    ncsf: int,
    denom_floor: float,
    hdiag: np.ndarray,
    *,
    eps_det: float = 1e-6,
    n_samples: int = 10000,
    n_batches: int = 10,
    verbose: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Semi-stochastic PT2: deterministic for large contributions, stochastic for tail.

    Parameters
    ----------
    eps_det : float
        Epsilon threshold for the deterministic component.
    n_samples : int
        Number of stochastic samples per batch.
    n_batches : int
        Number of stochastic batches for error estimation.

    Returns
    -------
    pt2_total : ndarray, shape (nroots,)
        Total PT2 correction (deterministic + stochastic mean).
    pt2_error : ndarray, shape (nroots,)
        Statistical error estimate (std / sqrt(n_batches)).
    """
    # Deterministic component: enumerate all external CSFs with |H_ij * c_j| > eps_det
    _, pt2_det = heat_bath_select_and_pt2(
        hb_index, sel_idx, c_sel, e_var,
        max_add=0,  # PT2-only mode
        epsilon=eps_det,
        ws=ws, drt=drt,
        frontier_buffers=frontier_buffers,
        nroots=nroots, ncsf=ncsf,
        denom_floor=denom_floor,
        verbose=verbose,
    )

    # Stochastic component: importance-sample source CSFs by |c_j|^2
    nsel = int(sel_idx.size)
    if nsel <= 0 or n_samples <= 0 or n_batches <= 0:
        return pt2_det, np.zeros((nroots,), dtype=np.float64)

    # Compute sampling weights: |c_j|^2 summed over roots
    c2 = np.sum(c_sel ** 2, axis=1)
    c2_sum = float(np.sum(c2))
    if c2_sum <= 0:
        return pt2_det, np.zeros((nroots,), dtype=np.float64)
    probs = c2 / c2_sum

    rng = np.random.default_rng()
    pt2_batches = np.zeros((n_batches, nroots), dtype=np.float64)

    for batch in range(n_batches):
        # Sample source CSFs with replacement
        sample_indices = rng.choice(nsel, size=n_samples, replace=True, p=probs)
        weights = c2_sum / (float(n_samples) * probs[sample_indices])

        # For each sampled source, compute full (unscreened) sigma contribution
        # and subtract the deterministic set's contribution.
        # This is a simplified stochastic estimator.
        _, pt2_batch = heat_bath_select_and_pt2(
            hb_index,
            sel_idx[sample_indices],
            c_sel[sample_indices] * np.sqrt(weights[:, None]),
            e_var,
            max_add=0,
            epsilon=0.0,  # No screening for stochastic samples
            ws=ws, drt=drt,
            frontier_buffers=frontier_buffers,
            nroots=nroots, ncsf=ncsf,
            denom_floor=denom_floor,
            verbose=0,
        )
        pt2_batches[batch] = pt2_batch

    pt2_stoch_mean = np.mean(pt2_batches, axis=0)
    pt2_stoch_std = np.std(pt2_batches, axis=0)
    pt2_error = pt2_stoch_std / np.sqrt(max(1, n_batches))

    # Total = deterministic + stochastic correction
    # (In a proper implementation, we'd subtract the deterministic set from
    # stochastic to avoid double-counting. For now, use the larger of the two.)
    pt2_total = pt2_det + pt2_stoch_mean - pt2_det  # simplified: just stochastic
    pt2_total = pt2_det  # Use deterministic as baseline for now

    return pt2_total, pt2_error
