from __future__ import annotations

import time
from typing import Any

import numpy as np


def resolve_kernel_nroots(
    *,
    requested_nroots: int | None,
    defaults: Any,
    ncsf: int,
) -> int:
    if requested_nroots is None:
        requested_nroots = int(getattr(defaults, "nroots", 1))
    nroots = int(requested_nroots)
    if nroots < 1:
        raise ValueError("nroots must be >= 1")
    if int(ncsf) < int(nroots):
        raise ValueError(f"nroots={int(nroots)} > ncsf={int(ncsf)}")
    return int(nroots)


def resolve_kernel_warm_start(
    *,
    ci0: Any,
    warm_state_enable: bool,
    warm_state_ci0_if_compatible_fn: Any,
    norb: int,
    nelec_total: int,
    twos: int,
    nroots: int,
    ncsf: int,
    orbsym: Any,
    wfnsym: int | None,
    ne_constraints: dict[int, tuple[int, int]] | None,
    matvec_backend: str,
    cas_metadata: dict[str, Any],
) -> dict[str, Any]:
    warm_applied = False
    warm_reason = "warm_start_disabled" if not warm_state_enable else "ci0_provided"
    ci0_out = ci0

    if ci0_out is None:
        if warm_state_enable:
            ci0_warm, warm_reason = warm_state_ci0_if_compatible_fn(
                norb=int(norb),
                nelec_total=int(nelec_total),
                twos=int(twos),
                nroots=int(nroots),
                ncsf=int(ncsf),
                orbsym=orbsym,
                wfnsym=wfnsym,
                ne_constraints=ne_constraints,
                matvec_backend=matvec_backend,
                cas_metadata=cas_metadata,
            )
            if ci0_warm is not None:
                ci0_out = ci0_warm
                warm_applied = True
        else:
            warm_reason = "warm_start_disabled"

    return {
        "ci0": ci0_out,
        "warm_applied": bool(warm_applied),
        "warm_reason": str(warm_reason),
    }


def build_kernel_dry_run_result(
    *,
    ci0: Any,
    nroots: int,
    ncsf: int,
    ecore: float,
    normalize_ci0_fn: Any,
) -> dict[str, Any]:
    if ci0 is None:
        ci = []
        for root in range(int(nroots)):
            v = np.zeros(int(ncsf), dtype=np.float64)
            v[int(root)] = 1.0
            ci.append(v)
    else:
        ci = normalize_ci0_fn(ci0, nroots=int(nroots), ncsf=int(ncsf))

    e = np.zeros(int(nroots), dtype=np.float64) + float(ecore)
    return {
        "e": np.asarray(e, dtype=np.float64),
        "ci": ci,
    }


def run_kernel_dense_eigh_fastpath(
    *,
    solver: Any,
    h1e: Any,
    eri: Any,
    norb: int,
    nelec: int | tuple[int, int],
    ncsf: int,
    nroots: int,
    ecore: float,
    max_out: int,
    orbsym: Any,
    wfnsym: int | None,
    ne_constraints: dict[int, tuple[int, int]] | None,
    drt_key: Any,
    warm_state_update: bool,
    nelec_total: int,
    twos: int,
    cas_metadata: dict[str, Any],
    warm_state_mo_coeff: Any,
    warm_state_mo_occ: Any,
    t_kernel0: float,
) -> dict[str, Any] | None:
    try:
        from asuka.integrals.df_integrals import DeviceDFMOIntegrals  # noqa: PLC0415
    except Exception:  # pragma: no cover
        DeviceDFMOIntegrals = ()  # type: ignore[assignment]
    try:
        from asuka.integrals.direct_integrals import DeviceDirectMOIntegrals  # noqa: PLC0415
    except Exception:  # pragma: no cover
        DeviceDirectMOIntegrals = ()  # type: ignore[assignment]

    if isinstance(eri, DeviceDFMOIntegrals) and getattr(eri, "eri_mat", None) is None:
        return None
    if isinstance(eri, DeviceDirectMOIntegrals):
        return None

    dense_thresh = int(getattr(solver, "dense_eigh_ncsf_threshold", 0))
    if dense_thresh <= 0 or int(ncsf) > dense_thresh:
        return None

    t_dense0 = time.perf_counter()
    hdiag_ps = None
    addr_full, h_full = solver.pspace(
        h1e,
        eri,
        norb,
        nelec,
        npsp=int(ncsf),
        max_out=int(max_out),
        hdiag=hdiag_ps,
        orbsym=orbsym,
        wfnsym=wfnsym,
        ne_constraints=ne_constraints,
    )
    addr_full = np.asarray(addr_full, dtype=np.int64).ravel()
    if int(addr_full.size) != int(ncsf):
        raise RuntimeError(
            "dense_eigh fast-path: pspace address size mismatch "
            f"(got {int(addr_full.size)}, expected {int(ncsf)})"
        )
    h_full = np.asarray(h_full, dtype=np.float64)
    h_full = 0.5 * (h_full + h_full.T)
    evals, evecs = np.linalg.eigh(h_full)
    order = np.argsort(np.asarray(evals, dtype=np.float64))[:int(nroots)]
    e = np.asarray(evals, dtype=np.float64)[order] + float(ecore)
    ci: list[np.ndarray] = []
    for col in order.tolist():
        v_sub = np.asarray(evecs[:, int(col)], dtype=np.float64).ravel()
        v_full = np.zeros((int(ncsf),), dtype=np.float64)
        v_full[addr_full] = v_sub
        ci.append(np.ascontiguousarray(v_full))

    if warm_state_update:
        solver._update_warm_state(
            ci=ci,
            norb=int(norb),
            nelec_total=int(nelec_total),
            twos=int(twos),
            nroots=int(nroots),
            ncsf=int(ncsf),
            orbsym=orbsym,
            wfnsym=wfnsym,
            ne_constraints=ne_constraints,
            cas_metadata=cas_metadata,
            mo_coeff=warm_state_mo_coeff,
            mo_occ=warm_state_mo_occ,
        )

    t_dense1 = time.perf_counter()
    return {
        "e": np.asarray(e, dtype=np.float64),
        "ci": [np.ascontiguousarray(v) for v in ci],
        "converged": np.ones((int(nroots),), dtype=np.bool_),
        "drt_key": drt_key,
        "dense_eigh_ncsf": int(ncsf),
        "dense_eigh_s": float(t_dense1 - t_dense0),
        "total_s": float(t_dense1 - float(t_kernel0)),
    }


def prepare_kernel_precompute_and_state_cache(
    *,
    precompute_epq: bool,
    drt: Any,
    matvec_backend: str,
    row_oracle_use_state_cache: bool,
    precompute_epq_actions_fn: Any,
    get_state_cache_fn: Any,
) -> Any:
    if bool(precompute_epq):
        precompute_epq_actions_fn(drt)

    state_cache = None
    if str(matvec_backend) == "row_oracle_df" and bool(row_oracle_use_state_cache):
        state_cache = get_state_cache_fn(drt)
    return state_cache


def normalize_row_screening(
    *,
    row_screening: Any,
    row_screening_type: Any,
) -> Any:
    if row_screening is not None and not isinstance(row_screening, row_screening_type):
        raise TypeError("row_screening must be a RowScreening or None")
    return row_screening


def maybe_restore_contract_eri(
    *,
    matvec_backend: str,
    eri: Any,
    norb: int,
    df_types: tuple[type, ...],
    restore_eri1_fn: Any,
) -> Any:
    """Best-effort one-time restore of sym=4 ERI for dense contract backend."""

    if str(matvec_backend) != "contract" or isinstance(eri, df_types):
        return eri

    try:
        eri_arr = np.asarray(eri)
        if eri_arr.ndim != 4:
            return restore_eri1_fn(eri, int(norb))
    except Exception:
        # Fall back to downstream restore logic if shape probing/restoration fails.
        return eri
    return eri


def build_cuda_hamiltonian_inputs(
    *,
    cp: Any,
    eri: Any,
    h1e: Any,
    norb: int,
    df_eri_mat_max_bytes: int,
    df_type: type,
    device_df_type: type,
    direct_device_type: type | tuple[type, ...] = (),
    restore_eri_4d_fn: Any,
) -> dict[str, Any]:
    """Build CUDA-side Hamiltonian inputs for matvec backends."""
    norb_i = int(norb)
    nops = int(norb_i) * int(norb_i)
    max_bytes = int(max(0, int(df_eri_mat_max_bytes)))

    eri_mat_d = None
    l_full_d = None
    direct_op_d = None
    if isinstance(eri, device_df_type):
        j_ps_d = cp.asarray(eri.j_ps, dtype=cp.float64)
        j_ps_d = cp.ascontiguousarray(j_ps_d)
        h1e_d = cp.asarray(np.asarray(h1e, dtype=np.float64), dtype=cp.float64)
        if eri.eri_mat is not None:
            eri_mat_d = cp.asarray(eri.eri_mat, dtype=cp.float64)
            eri_mat_d = cp.ascontiguousarray(eri_mat_d)
        elif eri.l_full is not None:
            l_full_d = cp.asarray(eri.l_full, dtype=cp.float64)
            l_full_d = cp.ascontiguousarray(l_full_d)
            need = int(nops) * int(nops) * int(np.dtype(np.float64).itemsize)
            if int(max_bytes) > 0 and int(need) <= int(max_bytes):
                eri_mat_d = cp.dot(l_full_d, l_full_d.T)
                eri_mat_d = cp.ascontiguousarray(eri_mat_d)
                l_full_d = None
        else:
            raise RuntimeError("DeviceDFMOIntegrals requires eri_mat or l_full for CUDA matvec")
        h_eff_d = h1e_d - 0.5 * j_ps_d
    elif isinstance(eri, direct_device_type):
        j_ps_src = getattr(eri, "j_ps_device", None)
        if j_ps_src is None:
            j_ps_src = eri.j_ps
        j_ps_d = cp.asarray(j_ps_src, dtype=cp.float64)
        j_ps_d = cp.ascontiguousarray(j_ps_d)
        h1e_d = cp.asarray(np.asarray(h1e, dtype=np.float64), dtype=cp.float64)
        h_eff_d = h1e_d - 0.5 * j_ps_d
        direct_op_d = eri
    else:
        if isinstance(eri, df_type):
            l_full_d = cp.asarray(eri.l_full, dtype=cp.float64)
            l_full_d = cp.ascontiguousarray(l_full_d)
            j_ps = np.asarray(eri.j_ps, dtype=np.float64, order="C")
            need = int(nops) * int(nops) * int(np.dtype(np.float64).itemsize)
            if int(max_bytes) > 0 and int(need) <= int(max_bytes):
                eri_mat_d = cp.dot(l_full_d, l_full_d.T)
                eri_mat_d = cp.ascontiguousarray(eri_mat_d)
                l_full_d = None
        else:
            eri4 = restore_eri_4d_fn(eri, int(norb_i)).astype(np.float64, copy=False)
            eri_mat_host = np.asarray(eri4.reshape(nops, nops), dtype=np.float64, order="C")
            j_ps = np.einsum("pqqs->ps", eri4).astype(np.float64, copy=False)

        h_eff = np.asarray(h1e, dtype=np.float64) - 0.5 * np.asarray(j_ps, dtype=np.float64)
        h_eff_d = cp.asarray(h_eff, dtype=cp.float64)
        if not isinstance(eri, df_type):
            eri_mat_d = cp.asarray(eri_mat_host, dtype=cp.float64)

    return {
        "eri_mat_d": eri_mat_d,
        "l_full_d": l_full_d,
        "direct_op_d": direct_op_d,
        "h_eff_d": h_eff_d,
    }


def autotune_cuda_max_g_mib_for_large_cas(
    *,
    max_g_mib: float,
    max_g_forced: bool,
    aggregate_offdiag: bool,
    ncsf: int,
    norb: int,
    matvec_cuda_dtype: str,
    eri_mat_present: bool,
    mem_hard_cap_gib: float,
    cuda_budget_free_bytes_fn: Any,
) -> float:
    """Best-effort auto-tune for CUDA matvec g-buffer size on large-CAS aggregate paths."""
    out = float(max_g_mib)
    if (
        bool(max_g_forced)
        or float(out) != 256.0
        or (not bool(aggregate_offdiag))
        or int(ncsf) < 1_000_000
        or int(norb) * int(norb) > 256
    ):
        return float(out)

    try:
        import cupy as cp  # type: ignore[import-not-found]

        free_b = cuda_budget_free_bytes_fn(cp, float(mem_hard_cap_gib))
        _free_raw_b, total_b = cp.cuda.runtime.memGetInfo()
        total_b = int(total_b)
    except Exception:
        free_b = None
        total_b = None

    w_itemsize = 4 if str(matvec_cuda_dtype) == "float32" else 8
    w_bytes = int(ncsf) * int(norb) * int(norb) * int(w_itemsize) if bool(eri_mat_present) else 0
    headroom = 4 * 1024 * 1024 * 1024
    if total_b is not None and int(total_b) <= 14 * 1024 * 1024 * 1024:
        cand_mibs = (512.0, 256.0)
    elif str(matvec_cuda_dtype) in ("float32", "mixed"):
        cand_mibs = (1024.0, 512.0, 256.0)
    else:
        cand_mibs = (2048.0, 1024.0, 512.0)

    if free_b is not None:
        for cand_mib in cand_mibs:
            cand_bytes = int(float(cand_mib) * 1024 * 1024)
            if int(free_b) >= int(w_bytes) + int(cand_bytes) + int(headroom):
                out = float(cand_mib)
                break
    return float(out)
