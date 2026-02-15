from __future__ import annotations

import contextlib
import numpy as np
import time
import warnings

from .basis_cart import BasisCartSoA
from .cart import ncart
from .eri_utils import expand_eri_packed_to_ordered, npair as npair_fn
from .shell_pairs import ShellPairs, build_shell_pairs_l_order
from .tasks import build_tasks_screened, build_tasks_screened_sorted_q, group_tasks_by_spab


def _detect_cpu_kernel_limits(*, default_lmax: int = 6) -> tuple[int, int]:
    """Best-effort discovery of compiled CPU ERI limits from the extension."""

    lmax = int(default_lmax)
    nroots_max = int(2 * lmax + 1)
    try:
        from . import _eri_rys_cpu as _ext  # noqa: PLC0415
    except Exception:
        return lmax, nroots_max

    limits_fn = getattr(_ext, "kernel_limits_cy", None)
    if limits_fn is None:
        return lmax, nroots_max
    try:
        limits = dict(limits_fn())
        lmax_i = int(limits.get("lmax", lmax))
        nroots_i = int(limits.get("nroots_max", 2 * lmax_i + 1))
    except Exception:
        return lmax, nroots_max
    if lmax_i < 0 or nroots_i < 1:
        return lmax, nroots_max
    return lmax_i, nroots_i


CPU_MAX_L, CPU_MAX_NROOTS = _detect_cpu_kernel_limits()


def _require_eri_cpu_ext():
    try:
        from . import _eri_rys_cpu as _ext  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "CPU ERI extension is not built. Build it with:\n"
            "  python -m asuka.cueri.build_cpu_ext build_ext --inplace"
        ) from e
    return _ext


def _optional_pair_coeff_cpu_ext():
    try:
        from . import _pair_coeff_cpu as _ext  # noqa: PLC0415
    except Exception:
        return None
    return _ext


def _resolve_effective_max_l(shell_l: np.ndarray, max_l: int | None) -> int:
    shell_l = np.asarray(shell_l, dtype=np.int32).ravel()
    shell_l_max = int(np.max(shell_l)) if int(shell_l.size) else 0
    if max_l is None:
        return shell_l_max
    max_l_i = int(max_l)
    if max_l_i < 0:
        raise ValueError("max_l must be >= 0")
    if shell_l_max > max_l_i:
        raise NotImplementedError("basis has shells with l > max_l")
    return max_l_i


def schwarz_shellpairs_cpu(
    ao_basis: BasisCartSoA,
    shell_pairs: ShellPairs,
    *,
    pair_tables=None,
    max_l: int | None = None,
    threads: int = 0,
    profile: dict | None = None,
) -> np.ndarray:
    """Compute exact Schwarz bounds Q_AB for AO shell pairs on the CPU.

    Evaluates the screening metric `Q_AB = sqrt(max((mn|mn)))` for all contracted
    Gaussian functions m in shell A and n in shell B. This upper bound is used for
    rigorous integral screening.

    Parameters
    ----------
    ao_basis : BasisCartSoA
        The atomic orbital basis set definition.
    shell_pairs : ShellPairs
        The list of shell pairs (A, B) to evaluate.
    pair_tables : object, optional
        Pre-computed pair data tables. If None, they are built on-the-fly.
    max_l : int | None, optional
        Maximum angular momentum to support. If None, inferred from `ao_basis`.
    threads : int, default=0
        Number of OpenMP threads to use. If 0, uses the environment default.
    profile : dict | None, optional
        Dictionary for collecting performance metrics.

    Returns
    -------
    np.ndarray
        Array of Schwarz bounds `Q_AB` with shape `(nsp,)`, aligned with `shell_pairs`.
    """

    if profile is not None:
        profile.clear()

    shell_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    _resolve_effective_max_l(shell_l, max_l)

    nsp = int(shell_pairs.sp_A.shape[0])
    if nsp == 0:
        return np.empty((0,), dtype=np.float64)

    _ext = _require_eri_cpu_ext()
    schwarz_sp = getattr(_ext, "schwarz_shellpairs_cart_sp_cy", None)
    if schwarz_sp is None:  # pragma: no cover
        raise RuntimeError("CPU ERI extension is missing Schwarz entry points; rebuild the extension")

    threads_i = int(threads)
    if threads_i < 0:
        raise ValueError("threads must be >= 0")
    if threads_i > 1 and hasattr(_ext, "openmp_enabled") and not bool(_ext.openmp_enabled()):
        warnings.warn(
            "threads>1 requested but asuka.cueri._eri_rys_cpu was built without OpenMP; "
            "rebuild with CUERI_USE_OPENMP=1 to enable parallelism",
            RuntimeWarning,
            stacklevel=2,
        )

    pt_prof = None
    if pair_tables is None:
        from .pair_tables_cpu import build_pair_tables_cpu  # local import to avoid import-time ext requirement

        pt_prof = {} if profile is not None else None
        pair_tables = build_pair_tables_cpu(ao_basis, shell_pairs, threads=threads_i, profile=pt_prof)

    pair_eta = np.asarray(pair_tables.pair_eta, dtype=np.float64, order="C")
    pair_Px = np.asarray(pair_tables.pair_Px, dtype=np.float64, order="C")
    pair_Py = np.asarray(pair_tables.pair_Py, dtype=np.float64, order="C")
    pair_Pz = np.asarray(pair_tables.pair_Pz, dtype=np.float64, order="C")
    pair_cK = np.asarray(pair_tables.pair_cK, dtype=np.float64, order="C")

    sp_A = np.asarray(shell_pairs.sp_A, dtype=np.int32, order="C")
    sp_B = np.asarray(shell_pairs.sp_B, dtype=np.int32, order="C")
    sp_pair_start = np.asarray(shell_pairs.sp_pair_start, dtype=np.int32, order="C")
    sp_npair = np.asarray(shell_pairs.sp_npair, dtype=np.int32, order="C")

    t0 = time.perf_counter() if profile is not None else 0.0
    out = schwarz_sp(
        np.asarray(ao_basis.shell_cxyz, dtype=np.float64, order="C"),
        np.asarray(ao_basis.shell_l, dtype=np.int32, order="C"),
        sp_A,
        sp_B,
        sp_pair_start,
        sp_npair,
        pair_eta,
        pair_Px,
        pair_Py,
        pair_Pz,
        pair_cK,
        threads_i,
    )
    if profile is not None:
        profile["t_schwarz_s"] = float(time.perf_counter() - t0)
        profile["threads"] = int(threads_i)
        profile["nsp"] = int(nsp)
        if pt_prof is not None:
            profile["pair_tables"] = pt_prof
    return out


def _build_pair_coeff_packed_tri(
    CA: np.ndarray,
    CB: np.ndarray,
    tri_p: np.ndarray,
    tri_q: np.ndarray,
    *,
    same_shell: bool,
) -> np.ndarray:
    CAp = CA[:, tri_p]  # (nA, npair)
    CBq = CB[:, tri_q]  # (nB, npair)
    K = CAp[:, None, :] * CBq[None, :, :]  # (nA, nB, npair)
    if not same_shell:
        CAq = CA[:, tri_q]
        CBp = CB[:, tri_p]
        K = K + CAq[:, None, :] * CBp[None, :, :]
    return K.reshape((int(CA.shape[0]) * int(CB.shape[0]), int(tri_p.size)))


def _build_pair_coeff_packed_batch_tri(
    C_active: np.ndarray,
    shell_ao_start: np.ndarray,
    shellA: np.ndarray,
    shellB: np.ndarray,
    nA: int,
    nB: int,
    tri_p: np.ndarray,
    tri_q: np.ndarray,
) -> np.ndarray:
    """Batched packed pair coefficient builder.

    Returns `K[t,(μν),pair]` with shape `(nt, nA*nB, npair)` for shell pairs (shellA[t], shellB[t]).
    """

    shellA = np.asarray(shellA, dtype=np.int32, order="C")
    shellB = np.asarray(shellB, dtype=np.int32, order="C")
    if shellA.shape != shellB.shape or shellA.ndim != 1:
        raise ValueError("shellA/shellB must be 1D arrays with identical shape")

    nt = int(shellA.shape[0])
    if nt == 0:
        return np.empty((0, int(nA) * int(nB), int(tri_p.size)), dtype=np.float64)

    shell_ao_start = np.asarray(shell_ao_start, dtype=np.int32).ravel()
    a0 = shell_ao_start[shellA].astype(np.int32, copy=False)
    b0 = shell_ao_start[shellB].astype(np.int32, copy=False)
    idxA = a0[:, None] + np.arange(int(nA), dtype=np.int32)[None, :]
    idxB = b0[:, None] + np.arange(int(nB), dtype=np.int32)[None, :]

    CA = C_active[idxA, :]  # (nt, nA, norb)
    CB = C_active[idxB, :]  # (nt, nB, norb)

    CAp = CA[:, :, tri_p]  # (nt, nA, npair)
    CBq = CB[:, :, tri_q]  # (nt, nB, npair)
    K = CAp[:, :, None, :] * CBq[:, None, :, :]  # (nt, nA, nB, npair)

    same = shellA == shellB
    if not bool(np.all(same)):
        CAq = CA[:, :, tri_q]
        CBp = CB[:, :, tri_p]
        mask = (~same)[:, None, None, None]
        K = K + (CAq[:, :, None, :] * CBp[:, None, :, :]) * mask

    return K.reshape((nt, int(nA) * int(nB), int(tri_p.size)))


def build_active_eri_packed_dense_cpu(
    ao_basis: BasisCartSoA,
    C_active: np.ndarray,  # (nao, norb), float64
    *,
    eps_ao: float = 0.0,
    eps_mo: float = 0.0,
    sp: ShellPairs | None = None,
    sp_Q: np.ndarray | None = None,
    pair_tables=None,
    max_l: int | None = None,
    max_tile_bytes: int = 256 << 20,
    threads: int = 0,
    blas_nthreads: int | None = None,
    profile: dict | None = None,
) -> np.ndarray:
    """Compute active-space ERIs as a packed pair matrix on the CPU.

    This function calculates (pq|rs) integrals for the active space defined by `C_active`.
    It uses a CPU-based Rys quadrature algorithm combined with an AB-grouping strategy
    for efficient contraction. The result is returned in a packed lower-triangular format.

    Parameters
    ----------
    ao_basis : BasisCartSoA
        The atomic orbital basis set definition.
    C_active : np.ndarray
        Active space MO coefficients. Shape: `(nao, norb)`.
    eps_ao : float, default=0.0
        Threshold for AO integral screening.
    eps_mo : float, default=0.0
        Threshold for MO integral screening (using AB-block estimates).
    sp : ShellPairs | None, optional
        The shell pair list. If None, built automatically.
    sp_Q : np.ndarray | None, optional
        Schwarz screening bounds. If None, computed if screening is enabled.
    pair_tables : object, optional
        Pre-computed pair data.
    max_l : int | None, optional
        Maximum angular momentum to support.
    max_tile_bytes : int, default=268435456
        Maximum size in bytes for temporary buffers.
    threads : int, default=0
        Number of threads for parallel execution.
    blas_nthreads : int | None, optional
        Number of threads for BLAS operations.
    profile : dict | None, optional
        Dictionary for performance metrics.

    Returns
    -------
    np.ndarray
        The ERI matrix of shape `(npair, npair)` where `npair = norb*(norb+1)/2`.
        Indices are packed canonical pairs `pq = p*(p+1)/2 + q` (p >= q).
    """

    if profile is not None:
        profile.clear()

    eps_ao_f = float(eps_ao)
    eps_mo_f = float(eps_mo)
    if eps_ao_f < 0.0 or eps_mo_f < 0.0:
        raise ValueError("eps_ao/eps_mo must be >= 0")

    shell_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    max_l_i = _resolve_effective_max_l(shell_l, max_l)

    C_active = np.asarray(C_active, dtype=np.float64, order="C")
    if C_active.ndim != 2:
        raise ValueError("C_active must be a 2D array with shape (nao, norb)")
    nao, norb = map(int, C_active.shape)

    npair = int(npair_fn(norb))
    tri_p, tri_q = np.tril_indices(norb)
    tri_p = np.asarray(tri_p, dtype=np.int32, order="C")
    tri_q = np.asarray(tri_q, dtype=np.int32, order="C")
    eri_packed = np.zeros((npair, npair), dtype=np.float64)

    if sp is None:
        sp = build_shell_pairs_l_order(ao_basis)

    nsp = int(sp.sp_A.shape[0])
    if nsp == 0 or norb == 0 or nao == 0:
        return eri_packed

    max_tile_bytes_i = int(max_tile_bytes)
    if max_tile_bytes_i <= 0:
        raise ValueError("max_tile_bytes must be > 0")

    threads_i = int(threads)
    if threads_i < 0:
        raise ValueError("threads must be >= 0")

    blas_nthreads_i: int | None
    if blas_nthreads is None:
        blas_nthreads_i = None
    else:
        blas_nthreads_i = int(blas_nthreads)
        if blas_nthreads_i < 1:
            raise ValueError("blas_nthreads must be >= 1")

    shell_ao_start = np.asarray(ao_basis.shell_ao_start, dtype=np.int32).ravel()

    # Optional BLAS thread control for the GEMM-heavy parts. This is important when
    # using OpenMP (`threads>1`) inside the ERI tile evaluator to avoid oversubscription.
    if blas_nthreads_i is not None:
        from asuka.cuguga.blas_threads import blas_thread_limit  # noqa: PLC0415

        blas_cm = blas_thread_limit(int(blas_nthreads_i))
    else:
        blas_cm = contextlib.nullcontext()

    t_total0 = time.perf_counter() if profile is not None else 0.0

    pt_prof = None
    if pair_tables is None:
        from .pair_tables_cpu import build_pair_tables_cpu  # local import to avoid import-time ext requirement

        pt_prof = {} if profile is not None else None
        pair_tables = build_pair_tables_cpu(ao_basis, sp, threads=threads_i, profile=pt_prof)

    if sp_Q is None:
        if eps_ao_f > 0.0 or eps_mo_f > 0.0:
            schwarz_prof = {} if profile is not None else None
            sp_Q = schwarz_shellpairs_cpu(
                ao_basis,
                sp,
                pair_tables=pair_tables,
                max_l=max_l,
                threads=threads_i,
                profile=schwarz_prof,
            )
            if profile is not None and schwarz_prof is not None:
                profile["schwarz"] = schwarz_prof
        else:
            sp_Q = np.ones((nsp,), dtype=np.float64)
    else:
        sp_Q = np.asarray(sp_Q, dtype=np.float64).ravel()
        if sp_Q.shape != (nsp,):
            raise ValueError(f"sp_Q must have shape (nsp,), got {sp_Q.shape} (nsp={nsp})")

    # Canonical screened task list (spCD <= spAB).
    t0 = time.perf_counter() if profile is not None else 0.0
    if eps_ao_f > 0.0:
        tasks = build_tasks_screened_sorted_q(sp_Q, eps=eps_ao_f)
    else:
        tasks = build_tasks_screened(sp_Q, eps=eps_ao_f)
    perm, ab_offsets = group_tasks_by_spab(tasks.task_spAB, nsp)
    task_cd = np.asarray(tasks.task_spCD[perm], dtype=np.int32, order="C")
    if profile is not None:
        profile["t_tasks_s"] = float(time.perf_counter() - t0)
        profile["ntasks"] = int(tasks.ntasks)

    _ext = _require_eri_cpu_ext()
    if threads_i > 1 and hasattr(_ext, "openmp_enabled") and not bool(_ext.openmp_enabled()):
        warnings.warn(
            "threads>1 requested but asuka.cueri._eri_rys_cpu was built without OpenMP; "
            "rebuild with CUERI_USE_OPENMP=1 to enable parallelism",
            RuntimeWarning,
            stacklevel=2,
        )
    eri_tile_sp = getattr(_ext, "eri_rys_tile_cart_sp_cy", None)
    eri_tile_sp_batch = getattr(_ext, "eri_rys_tile_cart_sp_batch_cy", None)
    if eri_tile_sp is None or eri_tile_sp_batch is None:  # pragma: no cover
        raise RuntimeError("CPU ERI extension is missing sp/pair-table entry points; rebuild the extension")

    _pair_ext = _optional_pair_coeff_cpu_ext()
    k_tri = getattr(_pair_ext, "build_pair_coeff_packed_tri_cy", None) if _pair_ext is not None else None
    k_batch = getattr(_pair_ext, "build_pair_coeff_packed_batch_tri_cy", None) if _pair_ext is not None else None

    pair_eta = np.asarray(pair_tables.pair_eta, dtype=np.float64, order="C")
    pair_Px = np.asarray(pair_tables.pair_Px, dtype=np.float64, order="C")
    pair_Py = np.asarray(pair_tables.pair_Py, dtype=np.float64, order="C")
    pair_Pz = np.asarray(pair_tables.pair_Pz, dtype=np.float64, order="C")
    pair_cK = np.asarray(pair_tables.pair_cK, dtype=np.float64, order="C")

    sp_A = np.asarray(sp.sp_A, dtype=np.int32, order="C")
    sp_B = np.asarray(sp.sp_B, dtype=np.int32, order="C")
    sp_pair_start = np.asarray(sp.sp_pair_start, dtype=np.int32, order="C")
    sp_npair = np.asarray(sp.sp_npair, dtype=np.int32, order="C")

    t_tile = 0.0
    t_gemm = 0.0
    t_kbuild = 0.0
    n_tile_calls = 0
    n_tile_tasks = 0
    n_skip_ab_ao_bound = 0
    n_skip_ab_mo_zero = 0
    n_skip_ab_mo_bound = 0

    M_sp: np.ndarray | None = None
    if eps_mo_f > 0.0:
        t0 = time.perf_counter() if profile is not None else 0.0
        n_shell = int(ao_basis.shell_cxyz.shape[0])
        M_shell = np.zeros((n_shell,), dtype=np.float64)
        for S in range(n_shell):
            start = int(ao_basis.shell_ao_start[S])
            n = int(ncart(int(shell_l[S])))
            if n <= 0:
                continue
            M_shell[S] = float(np.max(np.abs(C_active[start : start + n, :])))
        M_sp = M_shell[np.asarray(sp.sp_A, dtype=np.int32)] * M_shell[np.asarray(sp.sp_B, dtype=np.int32)]
        if profile is not None:
            profile["t_mo_screen_s"] = float(time.perf_counter() - t0)

    # Fast AB-level rejection bounds to avoid unnecessary per-task screening and
    # K_AB construction in heavily screened runs.
    q_max = 0.0
    qM_max: float | None = None
    if eps_ao_f > 0.0 or eps_mo_f > 0.0:
        q_max = float(np.max(sp_Q)) if int(sp_Q.shape[0]) else 0.0
    if eps_mo_f > 0.0 and M_sp is not None:
        qM_max = float(np.max(np.asarray(sp_Q, dtype=np.float64) * M_sp)) if int(nsp) else 0.0

    with blas_cm:
        for spAB in range(nsp):
            q_ab = float(sp_Q[spAB])
            if eps_ao_f > 0.0 and q_ab * q_max < eps_ao_f:
                n_skip_ab_ao_bound += 1
                continue

            if M_sp is not None and float(M_sp[spAB]) == 0.0:
                n_skip_ab_mo_zero += 1
                continue
            if eps_mo_f > 0.0 and M_sp is not None and qM_max is not None:
                if q_ab * float(M_sp[spAB]) * float(qM_max) < eps_mo_f:
                    n_skip_ab_mo_bound += 1
                    continue

            j0 = int(ab_offsets[spAB])
            j1 = int(ab_offsets[spAB + 1])
            if j0 == j1:
                continue

            spcd_all = task_cd[j0:j1]
            if spcd_all.size == 0:
                continue

            if eps_mo_f > 0.0:
                qprod = q_ab * sp_Q[spcd_all].astype(np.float64, copy=False)
                keep = qprod >= eps_ao_f
                keep &= qprod * float(M_sp[spAB]) * M_sp[spcd_all].astype(np.float64, copy=False) >= eps_mo_f
                if not bool(np.any(keep)):
                    continue
                spcd_keep = spcd_all[keep]
            else:
                spcd_keep = spcd_all

            A = int(sp.sp_A[spAB])
            B = int(sp.sp_B[spAB])
            la = int(shell_l[A])
            lb = int(shell_l[B])
            nA = int(ncart(la))
            nB = int(ncart(lb))
            nAB = nA * nB

            aoA = int(shell_ao_start[A])
            aoB = int(shell_ao_start[B])
            CA = C_active[aoA : aoA + nA, :]
            CB = C_active[aoB : aoB + nB, :]
            t0 = time.perf_counter() if profile is not None else 0.0
            if k_tri is None:
                K_AB = _build_pair_coeff_packed_tri(CA, CB, tri_p, tri_q, same_shell=(A == B))  # (nAB, npair)
            else:
                K_AB = k_tri(CA, CB, tri_p, tri_q, same_shell=(A == B), threads=threads_i)  # (nAB, npair)
            if profile is not None:
                t_kbuild += float(time.perf_counter() - t0)

            B_off = np.zeros((nAB, npair), dtype=np.float64)
            B_diag = np.zeros((nAB, npair), dtype=np.float64)

            # Diagonal term: (AB|AB)
            if bool(np.any(spcd_keep == spAB)):
                t0 = time.perf_counter() if profile is not None else 0.0
                tile_diag = eri_tile_sp(
                    ao_basis.shell_cxyz,
                    ao_basis.shell_l,
                    sp_A,
                    sp_B,
                    sp_pair_start,
                    sp_npair,
                    pair_eta,
                    pair_Px,
                    pair_Py,
                    pair_Pz,
                    pair_cK,
                    int(spAB),
                    int(spAB),
                )  # (nAB, nAB)
                if profile is not None:
                    t_tile += float(time.perf_counter() - t0)
                    n_tile_calls += 1
                    n_tile_tasks += 1

                t0 = time.perf_counter() if profile is not None else 0.0
                B_diag += tile_diag @ K_AB
                if profile is not None:
                    t_gemm += float(time.perf_counter() - t0)

            # Off-diagonal terms: batch by (lc,ld) so nCD is constant in each batch.
            spcd_off = spcd_keep[spcd_keep != spAB]
            if int(spcd_off.size) > 0:
                shellC_all = np.asarray(sp.sp_A[spcd_off], dtype=np.int32)
                shellD_all = np.asarray(sp.sp_B[spcd_off], dtype=np.int32)
                lc_all = shell_l[shellC_all]
                ld_all = shell_l[shellD_all]

                key_all = lc_all.astype(np.int32, copy=False) * int(max_l_i + 1) + ld_all.astype(np.int32, copy=False)
                for key in np.unique(key_all):
                    key_i = int(key)
                    lc0 = key_i // int(max_l_i + 1)
                    ld0 = key_i - lc0 * int(max_l_i + 1)

                    mask = key_all == key_i
                    spcd_grp = np.asarray(spcd_off[mask], dtype=np.int32, order="C")
                    shellC_grp = np.asarray(shellC_all[mask], dtype=np.int32, order="C")
                    shellD_grp = np.asarray(shellD_all[mask], dtype=np.int32, order="C")

                    nC = int(ncart(lc0))
                    nD = int(ncart(ld0))
                    nCD = int(nC * nD)

                    bytes_per_task = int(8 * npair * (nAB + nCD))
                    chunk_nt = int(max(1, max_tile_bytes_i // max(bytes_per_task, 1)))

                    for i0 in range(0, int(spcd_grp.size), chunk_nt):
                        i1 = min(int(spcd_grp.size), i0 + chunk_nt)
                        spcd = np.asarray(spcd_grp[i0:i1], dtype=np.int32, order="C")
                        shellC = np.asarray(shellC_grp[i0:i1], dtype=np.int32, order="C")
                        shellD = np.asarray(shellD_grp[i0:i1], dtype=np.int32, order="C")

                        t0 = time.perf_counter() if profile is not None else 0.0
                        tile = eri_tile_sp_batch(
                            ao_basis.shell_cxyz,
                            ao_basis.shell_l,
                            sp_A,
                            sp_B,
                            sp_pair_start,
                            sp_npair,
                            pair_eta,
                            pair_Px,
                            pair_Py,
                            pair_Pz,
                            pair_cK,
                            int(spAB),
                            spcd,
                            threads_i,
                        )  # (m, nAB, nCD)
                        if profile is not None:
                            t_tile += float(time.perf_counter() - t0)
                            n_tile_calls += 1
                            n_tile_tasks += int(spcd.shape[0])

                        t0 = time.perf_counter() if profile is not None else 0.0
                        if k_batch is None:
                            K_CD = _build_pair_coeff_packed_batch_tri(
                                C_active,
                                shell_ao_start,
                                shellC,
                                shellD,
                                nC,
                                nD,
                                tri_p,
                                tri_q,
                            )  # (m, nCD, npair)
                        else:
                            K_CD = k_batch(
                                C_active,
                                shell_ao_start,
                                shellC,
                                shellD,
                                nC,
                                nD,
                                tri_p,
                                tri_q,
                                threads=threads_i,
                            )  # (m, nCD, npair)
                        if profile is not None:
                            t_kbuild += float(time.perf_counter() - t0)

                        t0 = time.perf_counter() if profile is not None else 0.0
                        tmp = tile @ K_CD  # (m, nAB, npair)
                        B_off += tmp.sum(axis=0)
                        if profile is not None:
                            t_gemm += float(time.perf_counter() - t0)

            t0 = time.perf_counter() if profile is not None else 0.0
            contrib_off = K_AB.T @ B_off  # (npair, npair)
            eri_packed += contrib_off + contrib_off.T
            eri_packed += K_AB.T @ B_diag
            if profile is not None:
                t_gemm += float(time.perf_counter() - t0)

    if profile is not None:
        profile["t_total_s"] = float(time.perf_counter() - t_total0)
        profile["threads"] = int(threads_i)
        profile["blas_nthreads"] = None if blas_nthreads_i is None else int(blas_nthreads_i)
        profile["eps_ao"] = float(eps_ao_f)
        profile["eps_mo"] = float(eps_mo_f)
        profile["nao"] = int(nao)
        profile["norb"] = int(norb)
        profile["npair"] = int(npair)
        profile["nsp"] = int(nsp)
        if pt_prof is not None:
            profile["pair_tables"] = pt_prof
        profile["tile"] = {
            "t_tile_s": float(t_tile),
            "n_calls": int(n_tile_calls),
            "n_tasks": int(n_tile_tasks),
        }
        profile["screening"] = {
            "n_skip_ab_ao_bound": int(n_skip_ab_ao_bound),
            "n_skip_ab_mo_zero": int(n_skip_ab_mo_zero),
            "n_skip_ab_mo_bound": int(n_skip_ab_mo_bound),
        }
        profile["gemm"] = {"t_gemm_s": float(t_gemm)}
        profile["kbuild"] = {"t_kbuild_s": float(t_kbuild)}

    return eri_packed


def build_active_eri_mat_dense_cpu(
    ao_basis: BasisCartSoA,
    C_active: np.ndarray,
    *,
    eps_ao: float = 0.0,
    eps_mo: float = 0.0,
    sp: ShellPairs | None = None,
    sp_Q: np.ndarray | None = None,
    pair_tables=None,
    max_l: int | None = None,
    max_tile_bytes: int = 256 << 20,
    threads: int = 0,
    blas_nthreads: int | None = None,
    profile: dict | None = None,
) -> np.ndarray:
    """Compute active-space ERIs as a dense ordered-pair matrix on the CPU.

    Calculates (pq|rs) integrals and returns them in the ordered `pq, rs` flattened format.
    This is a wrapper around `build_active_eri_packed_dense_cpu` that unpacks the result.

    Parameters
    ----------
    ao_basis : BasisCartSoA
        The atomic orbital basis set definition.
    C_active : np.ndarray
        Active space MO coefficients. Shape: `(nao, norb)`.
    eps_ao : float, default=0.0
        AO screening threshold.
    eps_mo : float, default=0.0
        MO screening threshold.
    sp : ShellPairs | None, optional
        Shell pair list.
    sp_Q : np.ndarray | None, optional
        Schwarz screening bounds.
    pair_tables : object, optional
        Pre-computed pair data.
    max_l : int | None, optional
        Maximum angular momentum.
    max_tile_bytes : int, default=268435456
        Maximum temporary buffer size.
    threads : int, default=0
        Number of threads.
    blas_nthreads : int | None, optional
        Number of BLAS threads.
    profile : dict | None, optional
        Performance metrics dictionary.

    Returns
    -------
    np.ndarray
        The ERI matrix of shape `(norb*norb, norb*norb)`.
        Row index `pq = p * norb + q`, Col index `rs = r * norb + s`.
    """

    C_active = np.asarray(C_active, dtype=np.float64, order="C")
    norb = int(C_active.shape[1])
    eri_packed = build_active_eri_packed_dense_cpu(
        ao_basis,
        C_active,
        eps_ao=eps_ao,
        eps_mo=eps_mo,
        sp=sp,
        sp_Q=sp_Q,
        pair_tables=pair_tables,
        max_l=max_l,
        max_tile_bytes=max_tile_bytes,
        threads=threads,
        blas_nthreads=blas_nthreads,
        profile=profile,
    )
    return expand_eri_packed_to_ordered(eri_packed, norb)


__all__ = ["build_active_eri_mat_dense_cpu", "build_active_eri_packed_dense_cpu", "schwarz_shellpairs_cpu"]
