"""Heat-bath screened CIPSI selection + EN-PT2.

The production selector reuses the exact frontier-hash CSR -> ``g`` -> apply
pipeline, but screens the two-body generator tasks `(j, r, s)` using
``max_pq |(pq|rs)| * max_root |c_j| >= epsilon`` before building the CSR.
That preserves the exact ``epsilon = 0`` limit while still giving a
heat-bath-style task screen for finite ``epsilon``.

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
    """Build the legacy collapsed screened ``g_flat`` helper for one source CSF.

    This routine is kept as a small debugging/reference helper for the old
    collapsed-``g_flat`` approximation. The production HB selector no longer
    uses it, because the exact two-body path must preserve the intermediate
    `(j, k)` structure that is lost when off-diagonal `rs` terms are summed
    directly into one source-local ``g_flat[p, q]`` row.

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


def _sorted_desc_count(abs_sorted: np.ndarray, cutoff: float) -> int:
    """Count entries >= cutoff in a descending-sorted absolute-value array."""
    if abs_sorted.size == 0:
        return 0
    if cutoff <= 0.0:
        return int(abs_sorted.size)
    return int(np.searchsorted(-abs_sorted, -float(cutoff), side="right"))


def _ensure_hash_buffers(frontier_buffers: dict, cp: Any, cap: int, nroots: int) -> None:
    cap = int(cap)
    if int(frontier_buffers.get("hash_cap", 0)) == cap and frontier_buffers.get("hash_keys") is not None:
        return
    frontier_buffers["hash_cap"] = cap
    frontier_buffers["hash_keys"] = cp.empty((cap,), dtype=cp.int32)
    frontier_buffers["hash_vals"] = cp.empty((nroots, cap), dtype=cp.float64)
    frontier_buffers["hash_overflow"] = cp.empty((1,), dtype=cp.int32)
    frontier_buffers["out_idx"] = cp.empty((cap,), dtype=cp.int32)
    frontier_buffers["out_vals"] = cp.empty((nroots, cap), dtype=cp.float64)
    frontier_buffers["out_nnz"] = cp.empty((1,), dtype=cp.int32)


def _ensure_gdf_workspace(frontier_buffers: dict, *, nops: int, naux: int, nrows: int):
    from asuka.cuda.cuda_backend import Kernel3BuildGDFWorkspace  # noqa: PLC0415

    gdf_ws = frontier_buffers.get("hb_gdf_ws")
    max_rows = int(max(1, nrows))
    if gdf_ws is None or int(getattr(gdf_ws, "max_nrows", 0)) < max_rows:
        gdf_ws = Kernel3BuildGDFWorkspace(int(nops), int(naux), max_nrows=max_rows)
        frontier_buffers["hb_gdf_ws"] = gdf_ws
    return gdf_ws


def _heat_bath_select_screened_csr(
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
) -> tuple[list[int], np.ndarray]:
    """Exact screened HB path via screened `(j, r, s)` tasks and frontier-hash."""
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for HB-SCI") from e

    from asuka.cuda.cuda_backend import (  # noqa: PLC0415
        apply_g_flat_scatter_atomic_frontier_hash_many_roots_inplace_device,
        cipsi_frontier_hash_clear_inplace_device,
        cipsi_score_and_select_topk_from_hash_slots_inplace_device,
        kernel3_build_g_from_csr_eri_mat_range_inplace_device,
    )
    from asuka.cuguga.state_cache import get_state_cache  # noqa: PLC0415

    norb = hb_index.norb
    nops = norb * norb
    nsel = int(sel_idx.size)
    if nsel <= 0:
        return [], np.zeros((nroots,), dtype=np.float64)

    k25_ws = getattr(ws, "_k25_ws", None)
    row_j_buf = getattr(ws, "_csr_row_j", None)
    row_k_buf = getattr(ws, "_csr_row_k", None)
    indptr_buf = getattr(ws, "_csr_indptr", None)
    indices_buf = getattr(ws, "_csr_indices", None)
    data_buf = getattr(ws, "_csr_data", None)
    overflow_buf = getattr(ws, "_csr_overflow", None)
    if k25_ws is None or row_j_buf is None or row_k_buf is None or indptr_buf is None or indices_buf is None or data_buf is None or overflow_buf is None:
        raise RuntimeError("internal error: heat-bath selection requires Kernel25 workspace staging buffers")

    stream = cp.cuda.get_current_stream()
    sel_idx_i32 = np.asarray(sel_idx, dtype=np.int32)
    c_sel = np.asarray(c_sel, dtype=np.float64)
    max_cj = np.max(np.abs(c_sel), axis=1)
    active_pos = np.nonzero(max_cj > 0.0)[0]
    if active_pos.size == 0:
        return [], np.zeros((nroots,), dtype=np.float64)

    h1_abs = hb_index.h1_abs
    h1_signed = hb_index.h1_signed
    if "hb_h1_flat" not in frontier_buffers:
        frontier_buffers["hb_h1_flat"] = (
            hb_index.h1_pq[:, 0].astype(np.int64, copy=False) * norb
            + hb_index.h1_pq[:, 1].astype(np.int64, copy=False)
        )
    h1_flat = frontier_buffers["hb_h1_flat"]
    rs_flat = hb_index.rs_flat
    rs_max_v = hb_index.rs_max_v
    if "hb_offdiag_rs_r" not in frontier_buffers or "hb_offdiag_rs_s" not in frontier_buffers:
        rs_r_all = (rs_flat // norb).astype(np.int32, copy=False)
        rs_s_all = (rs_flat % norb).astype(np.int32, copy=False)
        offdiag_mask = rs_r_all != rs_s_all
        frontier_buffers["hb_offdiag_rs_r"] = rs_r_all[offdiag_mask]
        frontier_buffers["hb_offdiag_rs_s"] = rs_s_all[offdiag_mask]
        frontier_buffers["hb_offdiag_rs_max_v"] = rs_max_v[offdiag_mask]
    rs_r = frontier_buffers["hb_offdiag_rs_r"]
    rs_s = frontier_buffers["hb_offdiag_rs_s"]
    rs_offdiag_max_v = frontier_buffers["hb_offdiag_rs_max_v"]
    if "hb_diag_max_v" not in frontier_buffers:
        frontier_buffers["hb_diag_max_v"] = np.max(np.abs(hb_index.eri_diag_t), axis=1)
    diag_max_v = frontier_buffers["hb_diag_max_v"]
    state_cache = get_state_cache(drt)
    steps = np.asarray(state_cache.steps, dtype=np.int8)
    step_to_occ = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float64)

    l_full = getattr(ws, "l_full", None)
    eri_mat = getattr(ws, "eri_mat", None)
    if l_full is None and eri_mat is None:
        raise RuntimeError("internal error: workspace missing both eri_mat and l_full for HB-SCI")

    e_var_d = cp.ascontiguousarray(cp.asarray(e_var, dtype=cp.float64).ravel())
    selected_mask_d = frontier_buffers.get("selected_mask_d")
    hdiag_d = frontier_buffers.get("hdiag_d")
    if selected_mask_d is None or hdiag_d is None:
        raise RuntimeError("internal error: HB frontier buffers are missing selected_mask_d or hdiag_d")

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
    for _attempt in range(frontier_hash_max_retries):
        _ensure_hash_buffers(frontier_buffers, cp, cap, int(nroots))
        hash_keys = frontier_buffers["hash_keys"]
        hash_vals = frontier_buffers["hash_vals"]
        hash_overflow = frontier_buffers["hash_overflow"]

        cipsi_frontier_hash_clear_inplace_device(hash_keys, hash_vals, threads=256, stream=stream, sync=False)
        cp.cuda.runtime.memsetAsync(int(hash_overflow.data.ptr), 0, 4, int(stream.ptr))

        for sel_pos in active_pos.tolist():
            csf_j = int(sel_idx_i32[int(sel_pos)])
            coeff_row = np.asarray(c_sel[int(sel_pos)], dtype=np.float64)
            task_scale_row_d = cp.ascontiguousarray(cp.asarray(coeff_row.reshape(1, int(nroots)), dtype=cp.float64))
            cutoff = 0.0 if float(epsilon) <= 0.0 else float(epsilon) / float(max_cj[int(sel_pos)])

            nh1 = _sorted_desc_count(h1_abs, cutoff)
            if nh1 > 0:
                g_one_h = np.zeros((1, nops), dtype=np.float64)
                g_one_h[0, h1_flat[:nh1]] = h1_signed[:nh1]
                g_one_d = cp.asarray(g_one_h, dtype=cp.float64)
                task_csf_d = cp.asarray(np.asarray([csf_j], dtype=np.int32), dtype=cp.int32)
                apply_g_flat_scatter_atomic_frontier_hash_many_roots_inplace_device(
                    drt,
                    ws.drt_dev,
                    ws.state_dev,
                    task_csf_d,
                    g_one_d,
                    task_scale_task_major=task_scale_row_d,
                    hash_keys=hash_keys,
                    hash_vals=hash_vals,
                    selected_mask=selected_mask_d,
                    overflow=hash_overflow,
                    clear_overflow=False,
                    threads=256,
                    stream=stream,
                    sync=False,
                    check_overflow=False,
                )

            occ_j = step_to_occ[steps[csf_j].astype(np.int32, copy=False)]
            diag_keep = diag_max_v >= float(cutoff)
            if np.any(diag_keep):
                g_diag_h = np.dot(np.asarray(occ_j[diag_keep], dtype=np.float64), hb_index.eri_diag_t[diag_keep])
                g_diag_h = np.asarray(0.5 * g_diag_h, dtype=np.float64).reshape(1, nops)
                g_diag_d = cp.asarray(g_diag_h, dtype=cp.float64)
                task_csf_d = cp.asarray(np.asarray([csf_j], dtype=np.int32), dtype=cp.int32)
                apply_g_flat_scatter_atomic_frontier_hash_many_roots_inplace_device(
                    drt,
                    ws.drt_dev,
                    ws.state_dev,
                    task_csf_d,
                    g_diag_d,
                    task_scale_task_major=task_scale_row_d,
                    hash_keys=hash_keys,
                    hash_vals=hash_vals,
                    selected_mask=selected_mask_d,
                    overflow=hash_overflow,
                    clear_overflow=False,
                    threads=256,
                    stream=stream,
                    sync=False,
                    check_overflow=False,
                )

            nrs = _sorted_desc_count(rs_offdiag_max_v, cutoff)
            if nrs <= 0:
                continue

            task_csf_d = cp.asarray(np.full((nrs,), csf_j, dtype=np.int32), dtype=cp.int32)
            task_p_d = cp.asarray(rs_r[:nrs], dtype=cp.int32)
            task_q_d = cp.asarray(rs_s[:nrs], dtype=cp.int32)
            nrows, nnz, _nnz_in = k25_ws.build_from_tasks_deterministic_inplace_device(
                ws.drt_dev,
                ws.state_dev,
                task_csf_d,
                task_p_d,
                task_q_d,
                row_j_buf,
                row_k_buf,
                indptr_buf,
                indices_buf,
                data_buf,
                overflow_buf,
                int(getattr(ws, "threads_enum", 128)),
                bool(getattr(ws, "coalesce", False)),
                int(stream.ptr),
                True,
                True,
            )
            nrows = int(nrows)
            nnz = int(nnz)
            if nrows <= 0 or nnz <= 0:
                continue

            row_k_d = row_k_buf[:nrows]
            indptr_d = indptr_buf[: nrows + 1]
            indices_d = indices_buf[:nnz]
            data_d = data_buf[:nnz]
            g_rows_d = cp.empty((nrows, nops), dtype=cp.float64)

            if l_full is not None:
                gdf_ws = _ensure_gdf_workspace(
                    frontier_buffers,
                    nops=int(nops),
                    naux=int(l_full.shape[1]),
                    nrows=int(nrows),
                )
                gdf_ws.build_g_from_csr_l_full_range_inplace_device(
                    indptr_d,
                    indices_d,
                    data_d,
                    row_start=0,
                    nrows=int(nrows),
                    l_full=l_full,
                    g_out=g_rows_d,
                    threads=int(getattr(ws, "threads_g", 256)),
                    half=0.5,
                    stream=stream,
                    sync=False,
                )
            else:
                kernel3_build_g_from_csr_eri_mat_range_inplace_device(
                    indptr_d,
                    indices_d,
                    data_d,
                    row_start=0,
                    nrows=int(nrows),
                    eri_mat=eri_mat,
                    g_out=g_rows_d,
                    threads=int(getattr(ws, "threads_g", 256)),
                    half=0.5,
                    stream=stream,
                    sync=False,
                )

            # many-roots apply accepts a broadcast row shape (1, nroots),
            # avoiding materializing a repeated (nrows, nroots) coefficient matrix.
            task_scale_rows = task_scale_row_d
            apply_g_flat_scatter_atomic_frontier_hash_many_roots_inplace_device(
                drt,
                ws.drt_dev,
                ws.state_dev,
                row_k_d,
                g_rows_d,
                task_scale_task_major=task_scale_rows,
                hash_keys=hash_keys,
                hash_vals=hash_vals,
                selected_mask=selected_mask_d,
                overflow=hash_overflow,
                clear_overflow=False,
                threads=int(getattr(ws, "threads_apply", 256)),
                stream=stream,
                sync=False,
                check_overflow=False,
            )

        stream.synchronize()
        if int(hash_overflow.get()[0]) != 0:
            cap = min(cap << 1, 1 << 30)
            continue

        out_new_idx = cp.empty((max(1, max_add),), dtype=cp.int32)
        out_new_n = cp.empty((1,), dtype=cp.int32)
        out_pt2 = cp.empty((nroots,), dtype=cp.float64)
        cipsi_score_and_select_topk_from_hash_slots_inplace_device(
            hash_keys,
            hash_vals,
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
    3. Keep only generator pairs `(r,s)` with `max_pq |(pq|rs)| >= cutoff_j`
    4. Reuse the exact frontier-hash CSR -> `g` -> apply pipeline on that screened task set
    5. Extract + score + topk via the existing CIPSI kernels

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
        `"python"` / `"cuda"` / `"cuda_fused"` / `"auto"`.
        `"python"` and `"cuda"` use the exact screened-CSR reference path.
        `"cuda_fused"` uses the fused HB screen+apply kernel path.
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
        backend_s = "cuda"

    if backend_s not in ("python", "cuda", "cuda_fused"):
        raise ValueError("backend must be 'python', 'cuda', 'cuda_fused', or 'auto'")
    if backend_s == "cuda_fused":
        return _heat_bath_select_cuda(
            hb_index,
            sel_idx,
            c_sel,
            e_var,
            max_add,
            epsilon,
            ws,
            drt,
            frontier_buffers,
            nroots,
            ncsf,
            denom_floor,
            verbose=verbose,
        )
    return _heat_bath_select_python(
        hb_index,
        sel_idx,
        c_sel,
        e_var,
        max_add,
        epsilon,
        ws,
        drt,
        frontier_buffers,
        nroots,
        ncsf,
        denom_floor,
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
    """Compatibility wrapper for the exact screened-CSR implementation."""
    _ = verbose
    return _heat_bath_select_screened_csr(
        hb_index,
        sel_idx,
        c_sel,
        e_var,
        max_add,
        epsilon,
        ws,
        drt,
        frontier_buffers,
        nroots,
        ncsf,
        denom_floor,
    )


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
    """CUDA fused HB selector using the native screen+apply kernel."""
    _ = verbose
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for HB-SCI CUDA backend") from e

    from asuka.cuda.cuda_backend import (  # noqa: PLC0415
        cipsi_frontier_hash_clear_inplace_device,
        cipsi_score_and_select_topk_from_hash_slots_inplace_device,
        hb_screen_and_apply_inplace_device,
    )

    sel_idx_i32 = np.asarray(sel_idx, dtype=np.int32).ravel()
    nsel_i = int(sel_idx_i32.size)
    if nsel_i <= 0:
        return [], np.zeros((int(nroots),), dtype=np.float64)
    c_sel_h = np.asarray(c_sel, dtype=np.float64)
    if c_sel_h.ndim != 2 or c_sel_h.shape != (nsel_i, int(nroots)):
        raise ValueError("c_sel must have shape (nsel, nroots)")
    e_var_d = cp.ascontiguousarray(cp.asarray(e_var, dtype=cp.float64).ravel())
    if e_var_d.shape != (int(nroots),):
        raise ValueError("e_var must have shape (nroots,)")

    selected_mask_d = frontier_buffers.get("selected_mask_d")
    hdiag_d = frontier_buffers.get("hdiag_d")
    if selected_mask_d is None or hdiag_d is None:
        raise RuntimeError("internal error: HB frontier buffers are missing selected_mask_d or hdiag_d")

    sel_idx_d = cp.asarray(sel_idx_i32, dtype=cp.int32)
    c_sel_d = cp.ascontiguousarray(cp.asarray(c_sel_h, dtype=cp.float64))

    h1_pq_d = cp.ascontiguousarray(cp.asarray(hb_index.h1_pq, dtype=cp.int32))
    h1_abs_d = cp.ascontiguousarray(cp.asarray(hb_index.h1_abs, dtype=cp.float64).ravel())
    h1_signed_d = cp.ascontiguousarray(cp.asarray(hb_index.h1_signed, dtype=cp.float64).ravel())
    pq_ptr_d = cp.ascontiguousarray(cp.asarray(hb_index.pq_ptr, dtype=cp.int64).ravel())
    rs_idx_d = cp.ascontiguousarray(cp.asarray(hb_index.rs_idx, dtype=cp.int32).ravel())
    v_abs_d = cp.ascontiguousarray(cp.asarray(hb_index.v_abs, dtype=cp.float64).ravel())
    v_signed_d = cp.ascontiguousarray(cp.asarray(hb_index.v_signed, dtype=cp.float64).ravel())
    pq_max_v_d = cp.ascontiguousarray(cp.asarray(hb_index.pq_max_v, dtype=cp.float64).ravel())
    n_h1 = int(h1_abs_d.size)

    cap = int(frontier_buffers.get("hash_cap", 0))
    if cap <= 0:
        nops = int(hb_index.norb) * int(hb_index.norb)
        mult = min(256, max(32, nops // 4))
        target = int(min(ncsf, max(1 << 20, mult * nsel_i)))
        cap = 1
        while cap < target:
            cap <<= 1
        cap = max(1024, cap)
        while cap > int(ncsf) and cap > 1:
            cap >>= 1

    stream = cp.cuda.get_current_stream()
    frontier_hash_max_retries = int(frontier_buffers.get("max_retries", 8))
    for _attempt in range(frontier_hash_max_retries):
        _ensure_hash_buffers(frontier_buffers, cp, cap, int(nroots))
        hash_keys = frontier_buffers["hash_keys"]
        hash_vals = frontier_buffers["hash_vals"]
        hash_overflow = frontier_buffers["hash_overflow"]

        cipsi_frontier_hash_clear_inplace_device(hash_keys, hash_vals, threads=256, stream=stream, sync=False)
        cp.cuda.runtime.memsetAsync(int(hash_overflow.data.ptr), 0, 4, int(stream.ptr))

        for root in range(int(nroots)):
            hb_screen_and_apply_inplace_device(
                drt,
                ws.drt_dev,
                ws.state_dev,
                sel_idx_d,
                c_sel_d[:, int(root)],
                nsel=nsel_i,
                nroots=int(nroots),
                root=int(root),
                h1_pq=h1_pq_d,
                h1_abs=h1_abs_d,
                h1_signed=h1_signed_d,
                n_h1=n_h1,
                pq_ptr=pq_ptr_d,
                rs_idx=rs_idx_d,
                v_abs=v_abs_d,
                v_signed=v_signed_d,
                pq_max_v=pq_max_v_d,
                eps=float(epsilon),
                hash_keys=hash_keys,
                hash_vals=hash_vals,
                selected_mask=selected_mask_d,
                overflow=hash_overflow,
                threads=256,
                stream=stream,
                sync=False,
            )

        stream.synchronize()
        if int(hash_overflow.get()[0]) != 0:
            cap = min(cap << 1, 1 << 30)
            continue

        out_new_idx = cp.empty((max(1, int(max_add)),), dtype=cp.int32)
        out_new_n = cp.empty((1,), dtype=cp.int32)
        out_pt2 = cp.empty((int(nroots),), dtype=cp.float64)
        cipsi_score_and_select_topk_from_hash_slots_inplace_device(
            hash_keys,
            hash_vals,
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
    """Semi-stochastic PT2 placeholder.

    The deterministic HB path is implemented and validated. The stochastic tail
    estimator is not, so this routine fails loudly instead of silently returning
    an incorrect result.
    """
    _ = (
        hb_index,
        sel_idx,
        c_sel,
        e_var,
        ws,
        drt,
        frontier_buffers,
        nroots,
        ncsf,
        denom_floor,
        hdiag,
        eps_det,
        n_samples,
        n_batches,
        verbose,
    )
    raise NotImplementedError(
        "semistochastic_pt2 is not implemented yet; use heat_bath_select_and_pt2(..., max_add=0) for deterministic PT2"
    )
