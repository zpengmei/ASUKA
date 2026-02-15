from __future__ import annotations

"""Dense AO-ERI builders for HF.

Builds the full ordered AO-pair ERI matrix `eri_mat[pq, rs]` where:
  `pq = p * nao + q`, `rs = r * nao + s`.
"""

from dataclasses import dataclass
import os
import time
from typing import Any

import numpy as np

from asuka.integrals.int1e_cart import nao_cart_from_basis


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


_HF_DENSE_MEM_BUDGET_GIB_DEFAULT = _env_float("ASUKA_HF_DENSE_MEM_BUDGET_GIB", 8.0)


@dataclass(frozen=True)
class DenseAOERIResult:
    eri_mat: Any
    nao: int
    backend: str
    layout: str
    nbytes: int


def estimate_dense_eri_nbytes(nao: int, *, dtype=np.float64) -> int:
    """Estimated memory footprint for ordered AO-pair ERI matrix."""

    nao_i = int(nao)
    if nao_i < 0:
        raise ValueError("nao must be >= 0")
    itemsize = int(np.dtype(dtype).itemsize)
    return int(nao_i * nao_i * nao_i * nao_i * itemsize)


def _budget_bytes_from_gib(mem_budget_gib: float | None) -> int | None:
    gib = _HF_DENSE_MEM_BUDGET_GIB_DEFAULT if mem_budget_gib is None else float(mem_budget_gib)
    if gib <= 0.0:
        return None
    return int(gib * (1 << 30))


def _build_ao_eri_mat_dense_rys_cuda_direct(
    ao_basis,
    *,
    threads: int,
    max_tile_bytes: int,
    eps_ao: float,
    nao: int,
    profile: dict | None = None,
):
    """Build dense AO ERI matrix on CUDA by direct AO-pair scatter (no C=I transforms)."""

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for CUDA dense ERI build") from e

    from asuka.cueri.cart import ncart  # noqa: PLC0415
    from asuka.cueri import _cueri_cuda_ext as _ext  # noqa: PLC0415
    from asuka.cueri.gpu import (  # noqa: PLC0415
        CUDA_MAX_L,
        CUDA_MAX_NROOTS,
        build_pair_tables_ss_device,
        eri_rys_generic_device,
        has_cuda_ext,
        to_device_basis_ss,
        to_device_shell_pairs,
    )
    from asuka.cueri.shell_pairs import build_shell_pairs_l_order  # noqa: PLC0415
    from asuka.cueri.tasks import (  # noqa: PLC0415
        TaskList,
        build_tasks_screened,
        build_tasks_screened_sorted_q,
        decode_eri_class_id,
        group_tasks_by_class,
        with_task_class_id,
    )

    if profile is not None:
        profile.clear()

    if not has_cuda_ext():
        raise RuntimeError("cuERI CUDA extension not available; build via `python -m asuka.cueri.build_cuda_ext`")

    basis = ao_basis
    shell_l = np.asarray(basis.shell_l, dtype=np.int32).ravel()
    if shell_l.size == 0:
        return cp.zeros((nao * nao, nao * nao), dtype=cp.float64)
    if int(shell_l.max()) > CUDA_MAX_L:
        raise NotImplementedError(
            f"CUDA dense AO ERI currently supports only l<={CUDA_MAX_L} per shell (nroots<={CUDA_MAX_NROOTS})"
        )

    sp = build_shell_pairs_l_order(basis)
    nsp = int(sp.sp_A.shape[0])
    if nsp == 0:
        return cp.zeros((nao * nao, nao * nao), dtype=cp.float64)

    Q_np: np.ndarray
    eps_ao_f = float(eps_ao)
    if eps_ao_f > 0.0:
        from asuka.cueri.screening import schwarz_shellpairs_device  # noqa: PLC0415

        Q_dev = schwarz_shellpairs_device(
            basis,
            sp,
            threads=int(threads),
            max_tiles_bytes=int(max_tile_bytes),
        )
        Q_np = cp.asnumpy(Q_dev)
        tasks = build_tasks_screened_sorted_q(Q_np, eps=eps_ao_f)
    else:
        Q_np = np.ones((nsp,), dtype=np.float64)
        tasks = build_tasks_screened(Q_np, eps=0.0)
    if tasks.ntasks == 0:
        return cp.zeros((nao * nao, nao * nao), dtype=cp.float64)

    tasks = with_task_class_id(tasks, sp, shell_l)
    assert tasks.task_class_id is not None
    perm, class_ids, offsets = group_tasks_by_class(tasks.task_class_id)
    task_ab = tasks.task_spAB[perm]
    task_cd = tasks.task_spCD[perm]
    task_ab_dev = cp.ascontiguousarray(cp.asarray(task_ab, dtype=cp.int32))
    task_cd_dev = cp.ascontiguousarray(cp.asarray(task_cd, dtype=cp.int32))

    dbasis = to_device_basis_ss(basis)
    dsp = to_device_shell_pairs(sp)
    pair_tables = build_pair_tables_ss_device(dbasis, dsp, stream=None, threads=int(threads))

    n_pair = int(nao) * int(nao)
    eri_mat = cp.zeros((n_pair, n_pair), dtype=cp.float64)
    eri_mat_flat = eri_mat.ravel()
    sp_A_dev = cp.ascontiguousarray(cp.asarray(sp.sp_A, dtype=cp.int32))
    sp_B_dev = cp.ascontiguousarray(cp.asarray(sp.sp_B, dtype=cp.int32))
    shell_ao_start_np = np.asarray(basis.shell_ao_start, dtype=np.int32).ravel()
    shell_ao_start_dev = cp.ascontiguousarray(cp.asarray(shell_ao_start_np, dtype=cp.int32))
    stream_ptr = int(cp.cuda.get_current_stream().ptr)

    n_kernel_calls = 0
    n_scatter_calls = 0

    t0 = time.perf_counter()
    for g in range(int(class_ids.shape[0])):
        cid = int(class_ids[g])
        j0 = int(offsets[g])
        j1 = int(offsets[g + 1])
        if j1 <= j0:
            continue

        la, lb, lc, ld = decode_eri_class_id(cid)
        nA = int(ncart(int(la)))
        nB = int(ncart(int(lb)))
        nC = int(ncart(int(lc)))
        nD = int(ncart(int(ld)))
        nAB = int(nA) * int(nB)
        nCD = int(nC) * int(nD)
        bytes_per_task = int(nAB) * int(nCD) * 8
        chunk_ntasks = int(max(1, int(max_tile_bytes) // max(bytes_per_task, 1)))

        for i0 in range(j0, j1, chunk_ntasks):
            i1 = min(j1, i0 + chunk_ntasks)
            task_group = TaskList(
                task_spAB=np.asarray(task_ab[i0:i1], dtype=np.int32),
                task_spCD=np.asarray(task_cd[i0:i1], dtype=np.int32),
            )
            if task_group.ntasks == 0:
                continue
            raw = eri_rys_generic_device(
                task_group,
                dbasis,
                dsp,
                pair_tables,
                la=int(la),
                lb=int(lb),
                lc=int(lc),
                ld=int(ld),
                stream=None,
                threads=int(threads),
            )
            n_kernel_calls += 1
            _ext.scatter_eri_tiles_ordered_inplace_device(
                task_ab_dev[i0:i1],
                task_cd_dev[i0:i1],
                sp_A_dev,
                sp_B_dev,
                shell_ao_start_dev,
                int(nao),
                int(nA),
                int(nB),
                int(nC),
                int(nD),
                raw.ravel(),
                eri_mat_flat,
                int(threads),
                int(stream_ptr),
                False,
            )
            n_scatter_calls += 1

    cp.cuda.get_current_stream().synchronize()

    if profile is not None:
        profile["algorithm"] = "ao_pair_scatter_direct"
        profile["t_build_s"] = float(time.perf_counter() - t0)
        profile["nao"] = int(nao)
        profile["n_pair"] = int(n_pair)
        profile["nsp"] = int(nsp)
        profile["ntasks"] = int(tasks.ntasks)
        profile["threads"] = int(threads)
        profile["max_tile_bytes"] = int(max_tile_bytes)
        profile["eps_ao"] = float(eps_ao_f)
        profile["kernel_calls"] = int(n_kernel_calls)
        profile["scatter_calls"] = int(n_scatter_calls)

    return eri_mat


def build_ao_eri_dense(
    ao_basis,
    *,
    backend: str = "cuda",
    threads: int | None = None,
    max_tile_bytes: int = 256 << 20,
    eps_ao: float = 0.0,
    max_l: int | None = None,
    mem_budget_gib: float | None = None,
    profile: dict | None = None,
) -> DenseAOERIResult:
    """Build dense AO ERIs on CPU or CUDA.

    Parameters
    ----------
    ao_basis
        Packed AO basis object.
    backend
        `cpu` or `cuda`.
    threads
        Build threads. Defaults to `0` on CPU and `256` on CUDA.
    max_tile_bytes
        Tile/workspace cap forwarded to cuERI dense builders.
    eps_ao
        AO screening threshold.
    max_l
        Optional angular-momentum cap (currently enforced on CPU path only).
    mem_budget_gib
        Optional hard cap for ERI matrix allocation; `<=0` disables cap.
        If None, uses env `ASUKA_HF_DENSE_MEM_BUDGET_GIB` (default 8.0 GiB).
    """

    backend_s = str(backend).strip().lower()
    if backend_s not in {"cpu", "cuda"}:
        raise ValueError("backend must be 'cpu' or 'cuda'")

    if profile is not None:
        profile.clear()

    nao = int(nao_cart_from_basis(ao_basis))
    if nao <= 0:
        raise ValueError("AO basis has zero AO functions")
    est_nbytes = int(estimate_dense_eri_nbytes(nao, dtype=np.float64))
    budget_bytes = _budget_bytes_from_gib(mem_budget_gib)
    if budget_bytes is not None and est_nbytes > int(budget_bytes):
        est_gib = float(est_nbytes) / float(1 << 30)
        budget_gib = float(budget_bytes) / float(1 << 30)
        raise MemoryError(
            f"dense AO ERI requires ~{est_gib:.3f} GiB but dense_mem_budget_gib={budget_gib:.3f} GiB. "
            "Use df=True, reduce basis size, or raise dense_mem_budget_gib."
        )

    threads_i = int(0 if backend_s == "cpu" else 256) if threads is None else int(threads)
    max_tile_bytes_i = int(max_tile_bytes)
    if threads_i < 0:
        raise ValueError("threads must be >= 0")
    if max_tile_bytes_i <= 0:
        raise ValueError("max_tile_bytes must be > 0")
    eps_ao_f = float(eps_ao)
    if eps_ao_f < 0.0:
        raise ValueError("eps_ao must be >= 0")

    t0 = time.perf_counter()

    if backend_s == "cpu":
        from asuka.cueri.dense_cpu import build_active_eri_mat_dense_cpu  # noqa: PLC0415

        C_id = np.eye(nao, dtype=np.float64)
        eri_prof = profile.setdefault("dense_eri_cpu", {}) if profile is not None else None
        eri_mat = build_active_eri_mat_dense_cpu(
            ao_basis,
            C_id,
            eps_ao=eps_ao_f,
            max_l=max_l,
            threads=int(threads_i),
            max_tile_bytes=int(max_tile_bytes_i),
            profile=eri_prof,
        )
    else:
        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("backend='cuda' requires CuPy") from e
        from asuka.cueri.dense import build_active_eri_mat_dense_rys  # noqa: PLC0415

        try:
            free_bytes, _total_bytes = cp.cuda.Device().mem_info
        except Exception as e:
            raise RuntimeError("backend='cuda' requested but no usable CUDA device is available") from e
        required_bytes = int(est_nbytes + max_tile_bytes_i)
        if required_bytes > int(0.9 * int(free_bytes)):
            req_gib = float(required_bytes) / float(1 << 30)
            free_gib = float(free_bytes) / float(1 << 30)
            raise MemoryError(
                f"dense AO ERI build requires ~{req_gib:.3f} GiB but current GPU free memory is "
                f"{free_gib:.3f} GiB. Use df=True, reduce basis size, or switch backend='cpu'."
            )

        eri_prof = profile.setdefault("dense_eri_cuda", {}) if profile is not None else None
        strategy = str(os.environ.get("ASUKA_HF_DENSE_CUDA_AO_STRATEGY", "auto")).strip().lower()
        if strategy not in {"direct", "legacy", "auto"}:
            raise ValueError("ASUKA_HF_DENSE_CUDA_AO_STRATEGY must be one of: direct, legacy, auto")
        if strategy == "legacy":
            C_id = cp.eye(nao, dtype=cp.float64)
            eri_mat = build_active_eri_mat_dense_rys(
                ao_basis,
                C_id,
                threads=int(threads_i),
                max_tile_bytes=int(max_tile_bytes_i),
                eps_ao=eps_ao_f,
            )
            if eri_prof is not None:
                eri_prof["algorithm"] = "legacy_c_identity_transform"
        else:
            try:
                eri_mat = _build_ao_eri_mat_dense_rys_cuda_direct(
                    ao_basis,
                    threads=int(threads_i),
                    max_tile_bytes=int(max_tile_bytes_i),
                    eps_ao=float(eps_ao_f),
                    nao=int(nao),
                    profile=eri_prof,
                )
            except Exception as e:
                if strategy == "direct":
                    raise
                C_id = cp.eye(nao, dtype=cp.float64)
                eri_mat = build_active_eri_mat_dense_rys(
                    ao_basis,
                    C_id,
                    threads=int(threads_i),
                    max_tile_bytes=int(max_tile_bytes_i),
                    eps_ao=eps_ao_f,
                )
                if eri_prof is not None:
                    eri_prof["algorithm"] = "legacy_c_identity_transform_fallback"
                    eri_prof["fallback_reason"] = f"{type(e).__name__}: {e}"

    elapsed = float(time.perf_counter() - t0)
    if profile is not None:
        profile["backend"] = str(backend_s)
        profile["nao"] = int(nao)
        profile["eri_nbytes"] = int(est_nbytes)
        profile["dense_mem_budget_gib"] = (
            float(_HF_DENSE_MEM_BUDGET_GIB_DEFAULT) if mem_budget_gib is None else float(mem_budget_gib)
        )
        profile["threads"] = int(threads_i)
        profile["max_tile_bytes"] = int(max_tile_bytes_i)
        profile["eps_ao"] = float(eps_ao_f)
        profile["t_build_s"] = float(elapsed)

    return DenseAOERIResult(
        eri_mat=eri_mat,
        nao=int(nao),
        backend=str(backend_s),
        layout="ordered_pairs",
        nbytes=int(est_nbytes),
    )


__all__ = ["DenseAOERIResult", "build_ao_eri_dense", "estimate_dense_eri_nbytes"]
