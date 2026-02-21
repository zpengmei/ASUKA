"""MS/XMS CASPT2 effective Hamiltonian build on CUDA (SS-first, C1, FP64).

This provides a DF-friendly Heff path that does not require full `eri_mo`:
  - uses SS CUDA `row_dots_by_case_cuda` exported from each ket state's solve,
  - builds transition (TG1,TG2,TG3) on GPU via EPQ transpose-range apply-all,
  - contracts OpenMolcas HCOUP kernels on GPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from asuka.caspt2.result import CASPT2EnergyResult
from asuka.caspt2.superindex import SuperindexMap


def build_heff_cuda(
    nstates: int,
    ss_results: list[CASPT2EnergyResult],
    ci_vectors: list[np.ndarray],
    drt,
    smap: SuperindexMap,
    *,
    device: int | None = None,
    profile: dict[str, Any] | None = None,
    verbose: int = 0,
) -> np.ndarray:
    """Build Heff on GPU using stored row-dots and GPU transition TG1/TG2/TG3."""

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for Heff CUDA backend") from e

    if device is not None:
        cp.cuda.Device(int(device)).use()

    nstates = int(nstates)
    if len(ss_results) != nstates or len(ci_vectors) != nstates:
        raise ValueError("ss_results/ci_vectors length mismatch with nstates")

    # Extract device row_dots buffers from SS results.
    row_dots_by_state: list[list[Any]] = []
    for j in range(nstates):
        bd = ss_results[j].breakdown if isinstance(ss_results[j].breakdown, dict) else {}
        rd = bd.get("row_dots_by_case_cuda", None)
        if rd is None:
            raise ValueError(
                f"state {j}: missing breakdown['row_dots_by_case_cuda']; run SS with store_row_dots=True"
            )
        if not isinstance(rd, (list, tuple)) or len(rd) != 13:
            raise ValueError(f"state {j}: row_dots_by_case_cuda must be a length-13 list")
        row_dots_by_state.append(list(rd))

    # Build on device and copy once at the end to avoid per-pair synchronizations.
    heff_d = cp.zeros((nstates, nstates), dtype=cp.float64)
    for i in range(nstates):
        heff_d[i, i] = float(ss_results[i].e_tot)

    # Overlaps (host; small).
    ovl = np.zeros((nstates, nstates), dtype=np.float64)
    for i in range(nstates):
        for j in range(nstates):
            if i == j:
                continue
            ovl[i, j] = float(np.dot(ci_vectors[i], ci_vectors[j]))

    from asuka.cuda.rdm123_gpu import make_trans_rdm123_raw_cuda, reorder_dm123_molcas_trans_cuda  # noqa: PLC0415
    from asuka.caspt2.cuda.hcoup_cuda import hcoup_case_contribution_cuda  # noqa: PLC0415

    stream = cp.cuda.get_current_stream()

    # Off-diagonal: build transition TG tensors once for i<j.
    for i in range(nstates):
        for j in range(i + 1, nstates):
            if verbose >= 1:
                print(f"  Heff CUDA: building transition TG for pair ({i},{j})...")

            prof_ij: dict[str, Any] | None = {} if profile is not None else None

            ev0 = ev1 = ev2 = None
            if prof_ij is not None:
                ev0 = cp.cuda.Event()
                ev1 = cp.cuda.Event()
                ev2 = cp.cuda.Event()
                ev0.record(stream)

            dm1_raw, dm2_raw, dm3_raw = make_trans_rdm123_raw_cuda(
                drt,
                ci_vectors[i],
                ci_vectors[j],
                device=device,
                profile=prof_ij,
            )
            tg1_ij, tg2_ij, tg3_ij = reorder_dm123_molcas_trans_cuda(
                dm1_raw, dm2_raw, dm3_raw, inplace=False, profile=prof_ij
            )

            # Reverse direction mapping for transition TG tensors:
            #   <j|E_pq|i> = <i|E_qp|j>
            # and similarly for higher-order TG tensors.
            tg1_ji = tg1_ij.T
            tg2_ji = tg2_ij.transpose(3, 2, 1, 0)
            tg3_ji = tg3_ij.transpose(5, 4, 3, 2, 1, 0)

            if prof_ij is not None and ev1 is not None:
                ev1.record(stream)

            coup_ij = cp.float64(0.0)
            coup_ji = cp.float64(0.0)

            for case in range(1, 14):
                rd_ket_j = row_dots_by_state[j][case - 1]
                rd_ket_i = row_dots_by_state[i][case - 1]
                if int(getattr(rd_ket_j, "size", 0)) != 0:
                    coup_ij = coup_ij + hcoup_case_contribution_cuda(
                        case, smap, rd_ket_j, tg1_ij, tg2_ij, tg3_ij, ovl=float(ovl[i, j])
                    )
                if int(getattr(rd_ket_i, "size", 0)) != 0:
                    coup_ji = coup_ji + hcoup_case_contribution_cuda(
                        case, smap, rd_ket_i, tg1_ji, tg2_ji, tg3_ji, ovl=float(ovl[j, i])
                    )

            heff_d[i, j] = coup_ij
            heff_d[j, i] = coup_ji

            if prof_ij is not None and ev2 is not None and ev0 is not None and ev1 is not None:
                ev2.record(stream)
                try:
                    stream.synchronize()
                except Exception:
                    pass
                prof_ij["heff_tg_s"] = float(cp.cuda.get_elapsed_time(ev0, ev1) * 1e-3)
                prof_ij["heff_hcoup_s"] = float(cp.cuda.get_elapsed_time(ev1, ev2) * 1e-3)

            if profile is not None and prof_ij is not None:
                profile[f"tg_pair_{i}_{j}"] = prof_ij

    return np.asarray(cp.asnumpy(heff_d), dtype=np.float64, order="C")
