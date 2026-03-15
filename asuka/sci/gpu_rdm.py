"""GPU-resident sparse RDM for selected CI wavefunctions.

Computes dm1 (norb, norb) and dm2 (norb, norb, norb, norb) entirely on GPU
using the bucketed pairwise GUGA T-matrix kernel:

    T[pq, i] = sum_j E_pq[i,j] * c[j]   (shape norb^2 x nsel, GPU)
    dm1 = (T @ c).reshape(norb, norb).T
    gram = T @ T.T
    dm2[p,q,r,s] = gram[q*norb+p, r*norb+s] - delta_qr * dm1[p,s]

No large H->D or D->H transfers: all data stays on GPU throughout.
"""

from __future__ import annotations

import numpy as np


def make_rdm12_gpu(
    drt,
    drt_dev,
    sel_idx,
    ci_sel,
    cp,
    *,
    threads: int = 256,
) -> tuple:
    """Compute (dm1, dm2) from sparse CI on GPU.

    Parameters
    ----------
    drt : DRT
        CPU-side DRT object.
    drt_dev : DeviceDRT
        GPU-resident DRT (from make_device_drt).
    sel_idx : array-like, int64
        Global CSF indices of selected states (CPU or GPU).
    ci_sel : array-like, float64
        CI coefficients for selected states (CPU or GPU).
    cp : module
        CuPy module.

    Returns
    -------
    dm1 : cupy array, shape (norb, norb)
    dm2 : cupy array, shape (norb, norb, norb, norb)
    """
    from asuka.cuda.cuda_backend import (
        pairwise_build_bucket_data,
        pairwise_materialize_u64_device,
        pairwise_T_matrix_bucketed_u64_device,
    )

    norb = int(drt.norb)
    nsel = int(np.asarray(sel_idx).size)

    # Transfer to GPU if needed
    sel_u64_d = cp.ascontiguousarray(
        cp.asarray(np.asarray(sel_idx, dtype=np.int64).ravel()).astype(cp.uint64)
    )
    ci_sel_d = cp.ascontiguousarray(
        cp.asarray(np.asarray(ci_sel, dtype=np.float64).ravel())
    )

    # Materialize CSF data on GPU
    materialized = pairwise_materialize_u64_device(
        drt, drt_dev, sel_u64_d, nsel, cp, threads=threads, sync=False,
    )
    steps_all, nodes_all, occ_all, b_all = materialized

    # Build bucket data and sort by occupation key
    bucket_data_full = pairwise_build_bucket_data(occ_all, norb, cp)
    sort_perm_d = cp.ascontiguousarray(
        bucket_data_full["sort_perm"].astype(cp.int32, copy=False)
    )
    inv_perm_d = cp.ascontiguousarray(
        bucket_data_full["inv_perm"].astype(cp.int32, copy=False)
    )
    materialized_sorted = (
        cp.ascontiguousarray(steps_all[sort_perm_d]),
        cp.ascontiguousarray(nodes_all[sort_perm_d]),
        cp.ascontiguousarray(occ_all[sort_perm_d]),
        cp.ascontiguousarray(b_all[sort_perm_d]),
    )
    ci_sorted_d = cp.ascontiguousarray(ci_sel_d[sort_perm_d])
    bucket_data = {
        "csf_to_bucket": cp.ascontiguousarray(bucket_data_full["csf_to_bucket"]),
        "target_offsets_1b": cp.ascontiguousarray(bucket_data_full["target_offsets_1b"]),
        "target_list_1b": cp.ascontiguousarray(bucket_data_full["target_list_1b"]),
    }

    # Compute T matrix on GPU: T[pq, i_sorted] = E_pq|c>[i_sorted]
    T_sorted_d = pairwise_T_matrix_bucketed_u64_device(
        drt, drt_dev,
        materialized_sorted,
        bucket_data,
        ci_sorted_d,
        nsel, cp,
        threads=threads, sync=True,
    )

    # Restore original CSF ordering in T
    T_d = cp.ascontiguousarray(T_sorted_d[:, inv_perm_d])  # (norb^2, nsel)

    # dm1 = (T @ c).reshape(norb, norb).T
    dm1_flat = T_d @ ci_sel_d  # (norb^2,)
    dm1 = dm1_flat.reshape(norb, norb).T  # (norb, norb)

    # dm2 via gram: gram0[pq, rs] = T[pq,:] @ T[rs,:]
    gram0 = T_d @ T_d.T  # (norb^2, norb^2)
    # swap rows: gram[pq, rs] = gram0[qp, rs] so dm2[p,q,r,s] = T[qp] @ T[rs]
    swap = cp.arange(norb * norb, dtype=cp.int32).reshape(norb, norb).T.ravel()
    gram = gram0[swap]  # (norb^2, norb^2)
    dm2 = gram.reshape(norb, norb, norb, norb).copy()

    # Correction: dm2[p,q,q,s] -= dm1[s,p] for all p,q,s
    # = dm2[:, q, q, :] -= dm1.T for each q
    for q in range(norb):
        dm2[:, q, q, :] -= dm1.T

    return dm1, dm2
