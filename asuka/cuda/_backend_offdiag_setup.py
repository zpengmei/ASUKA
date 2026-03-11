from __future__ import annotations

import os
from typing import Any

import numpy as np


def resolve_w_offdiag_prefer_blocked(
    *,
    dtype_is_float32: bool,
    ncsf: int,
    epq_table_present: bool,
    use_epq_table: bool,
    eri_mat_present: bool,
) -> bool:
    if not bool(eri_mat_present):
        return True
    return bool(
        bool(dtype_is_float32)
        and int(ncsf) >= 1_000_000
        and (bool(epq_table_present) or (not bool(use_epq_table)))
    )


def init_sym_pair_setup(
    *,
    cp,
    aggregate_offdiag_k: bool,
    has_sym_pair_pack_device: bool,
    norb: int,
    nops: int,
    dtype: Any,
    kernel3_workspace_ctor: Any,
) -> dict[str, Any]:
    out = {
        "_sym_pair_pair_pq": None,
        "_sym_pair_pair_qp": None,
        "_sym_pair_full_to_pair": None,
        "_sym_pair_npair": 0,
        "_sym_pair_eri_pair": None,
        "_sym_pair_eri_pair_src_id": None,
        "_sym_pair_l_pair": None,
        "_sym_pair_l_pair_src_id": None,
        "_sym_pair_w_pair": None,
        "_sym_pair_g_pair": None,
        "_sym_pair_gemm_ws": None,
    }

    sym_pair_min_norb = int(os.getenv("CUGUGA_SYM_PAIR_MIN_NORB", "10"))
    if (not bool(aggregate_offdiag_k)) or (not bool(has_sym_pair_pack_device)) or int(norb) < int(sym_pair_min_norb):
        return out

    norb_i = int(norb)
    npair = norb_i * (norb_i + 1) // 2
    pair_pq_h: list[int] = []
    pair_qp_h: list[int] = []
    full_to_pair_h = np.empty(int(nops), dtype=np.int32)
    u = 0
    for p in range(norb_i):
        for q in range(p, norb_i):
            pq = p * norb_i + q
            qp = q * norb_i + p
            pair_pq_h.append(pq)
            pair_qp_h.append(qp)
            full_to_pair_h[pq] = u
            full_to_pair_h[qp] = u
            u += 1

    out["_sym_pair_npair"] = int(npair)
    out["_sym_pair_pair_pq"] = cp.asarray(np.array(pair_pq_h, dtype=np.int32))
    out["_sym_pair_pair_qp"] = cp.asarray(np.array(pair_qp_h, dtype=np.int32))
    out["_sym_pair_full_to_pair"] = cp.asarray(full_to_pair_h)

    sp_backend = os.getenv("CUGUGA_SYM_PAIR_GEMM_BACKEND", "cublaslt_fp64")
    try:
        out["_sym_pair_gemm_ws"] = kernel3_workspace_ctor(
            int(npair),
            max_nrows=1,
            dtype=dtype,
            gemm_backend=str(sp_backend),
        )
    except Exception:
        try:
            out["_sym_pair_gemm_ws"] = kernel3_workspace_ctor(
                int(npair),
                max_nrows=1,
                dtype=dtype,
                gemm_backend="gemmex_fp64",
            )
        except Exception:
            out["_sym_pair_gemm_ws"] = None

    return out
