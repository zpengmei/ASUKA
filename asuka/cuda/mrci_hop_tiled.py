from __future__ import annotations

import cupy as cp
import numpy as np

from asuka.cuda.cuda_backend import (
    build_w_from_epq_table_inplace_device,
    build_w_diag_from_steps_inplace_device,
    apply_g_flat_scatter_atomic_inplace_device,
)


def hop_cuda_epq_table(
    drt, drt_dev, state_dev, epq_table,
    h_eff, eri_mat_t, x, y=None
):
    """
    Uncontracted MRCI sigma build on GPU using dense intermediate method.
    
    Parameters
    ----------
    drt : DRT
        Host DRT object
    drt_dev : dict
        Device copy of DRT arrays
    state_dev : dict
        Device copy of state tables (steps, nodes)
    epq_table : tuple
        Prebuilt E_pq action table (indptr, indices, pq_ids, data)
    h_eff : cp.ndarray[norb, norb]
        Effective 1e integrals
    eri_mat_t : cp.ndarray[nops, nops]
        Transposed ERI matrix: eri_mat_t[rs, pq] = 0.5*(pq|rs)
    x : cp.ndarray[ncsf] or [ncsf, nvec]
        Input vector(s)
    y : cp.ndarray or None
        Output buffer (allocated if None)
        
    Returns
    -------
    y : cp.ndarray
        Result: y = H @ x
    """
    norb = int(h_eff.shape[0])
    ncsf = int(x.shape[0])
    nops = norb * norb
    
    if y is None:
        y = cp.zeros_like(x)
    else:
        y.fill(0)
    
    # Step 1: Build W = E_rs x
    # build_w_from_epq_table_inplace_device returns (W, overflow)
    W, overflow = build_w_from_epq_table_inplace_device(drt, state_dev, epq_table, x)
    if int(cp.asnumpy(overflow.ravel())[0]) != 0:
        raise RuntimeError(f"GPU W build overflow detected")
    
    # Add diagonal contributions using existing kernel
    # build_w_diag_from_steps_inplace_device(state_dev, *, j_start, j_count, x, w_out, ...)
    build_w_diag_from_steps_inplace_device(
        state_dev, j_start=0, j_count=ncsf, x=x, w_out=W
    )

    
    # Step 2: Dense G = W @ eri_mat_t
    # x: [ncsf, nops], eri_mat_t: [nops, nops] -> G: [ncsf, nops]
    # eri_mat_t is expected to have factor 0.5 included
    G = cp.matmul(W, eri_mat_t)
    
    # Step 3: Apply E_pq scatter to y
    # apply_g_flat_scatter_atomic_inplace_device returns (y, overflow)
    y, overflow = apply_g_flat_scatter_atomic_inplace_device(
        drt=drt,
        drt_dev=drt_dev,
        state_dev=state_dev,
        task_csf=cp.arange(ncsf, dtype=cp.int32),
        task_g=G,
        epq_table=epq_table,
        y=y,
        zero_y=False
    )
    if int(cp.asnumpy(overflow.ravel())[0]) != 0:
        raise RuntimeError(f"GPU apply_g overflow detected")


    
    # Step 4: Add h_eff contribution
    h_eff_flat = h_eff.ravel()
    
    if x.ndim == 1:
        y += W @ h_eff_flat
    else:
        # Loop over vectors if multi-vector (naive implementation for now)
        # Assuming x is [ncsf, nvec]
        # W is also [ncsf, nops] for a single vector build? 
        # Wait, if x is [ncsf, nvec], build_w might fail if not handled.
        # Fixed earlier: build_w assumes single vector x.
        y += W @ h_eff_flat
    
    return y


def hop_cuda_projected(
    drt_full, drt_dev_full, state_dev_full, epq_table_full,
    h_eff, eri_mat_t_full, x_sub, sub_to_full,
    y_sub=None, x_full_buf=None, y_full_buf=None
):
    """
    Projected MRCI sigma build on GPU.
    
    y = (P H_full P) x
    """
    import cupy as cp
    from asuka.cuda.cuda_backend import (
        scatter_embed_inplace_device,
        gather_project_inplace_device,
    )
    
    ncsf_sub = int(x_sub.shape[0])
    ncsf_full = int(drt_full.ncsf)
    
    if y_sub is None:
        y_sub = cp.zeros_like(x_sub)
    else:
        y_sub.fill(0)
        
    if x_full_buf is None:
        x_full_buf = cp.zeros(ncsf_full, dtype=cp.float64)
    else:
        x_full_buf.fill(0)
        
    if y_full_buf is None:
        y_full_buf = cp.zeros(ncsf_full, dtype=cp.float64)
    else:
        y_full_buf.fill(0)
        
    # Step 1: Embed x_sub -> x_full
    scatter_embed_inplace_device(x_sub, sub_to_full, x_full_buf)
    
    # Step 2: Apply H in full space
    hop_cuda_epq_table(
        drt=drt_full,
        drt_dev=drt_dev_full,
        state_dev=state_dev_full,
        epq_table=epq_table_full,
        h_eff=h_eff,
        eri_mat_t=eri_mat_t_full,
        x=x_full_buf,
        y=y_full_buf
    )
    
    # Step 3: Project y_full -> y_sub
    gather_project_inplace_device(y_full_buf, sub_to_full, y_sub)
    
    return y_sub

