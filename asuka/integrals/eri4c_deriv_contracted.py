from __future__ import annotations

from dataclasses import dataclass
import weakref

import numpy as np

from asuka.cueri.basis_cart import BasisCartSoA
from asuka.cueri.pair_tables_cpu import PairTablesCPU
from asuka.cueri.shell_pairs import ShellPairs


def _require_eri_cpu_ext():
    try:
        from asuka.cueri import _eri_rys_cpu as _ext  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "CPU ERI extension is not built. Build it with:\n"
            "  python -m asuka.cueri.build_cpu_ext build_ext --inplace"
        ) from e
    return _ext


@dataclass(frozen=True)
class ERI4cDerivContractionCPUContext:
    shell_cxyz: np.ndarray  # float64, (nShell,3)
    shell_l: np.ndarray  # int32, (nShell,)
    shell_prim_start: np.ndarray  # int32, (nShell,)
    shell_nprim: np.ndarray  # int32, (nShell,)
    prim_exp: np.ndarray  # float64, (nPrim,)

    sp_A: np.ndarray  # int32, (nSP,)
    sp_B: np.ndarray  # int32, (nSP,)
    sp_pair_start: np.ndarray  # int32, (nSP+1,)
    sp_npair: np.ndarray  # int32, (nSP,)

    pair_eta: np.ndarray  # float64, (total_pair_prims,)
    pair_Px: np.ndarray  # float64, (total_pair_prims,)
    pair_Py: np.ndarray  # float64, (total_pair_prims,)
    pair_Pz: np.ndarray  # float64, (total_pair_prims,)
    pair_cK: np.ndarray  # float64, (total_pair_prims,)


@dataclass(frozen=True)
class ERI4cDerivContractionCUDAContext:
    dsp: object
    dbasis: object
    dpt: object
    device_id: int


_CUDA_CTX_CACHE_MAX = 8
_cuda_ctx_cache: dict[tuple[int, int, int], tuple[weakref.ref, weakref.ref, ERI4cDerivContractionCUDAContext]] = {}


def make_eri4c_deriv_contraction_cuda_context(
    basis: BasisCartSoA,
    shell_pairs: ShellPairs,
    *,
    threads: int = 256,
    stream=None,
) -> ERI4cDerivContractionCUDAContext:
    """Build (and cache) GPU-resident basis/shell-pair/pair-table objects for contracted 4c derivative kernels."""

    try:
        import cupy as cp  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CUDA contracted-derivative context requires CuPy") from e

    from asuka.cueri import gpu as cueri_gpu  # noqa: PLC0415

    threads_i = int(threads)
    if threads_i <= 0:
        raise ValueError("threads must be > 0 for CUDA context build")

    dev_id = int(cp.cuda.runtime.getDevice())
    key = (id(basis), id(shell_pairs), dev_id)

    hit = _cuda_ctx_cache.get(key)
    if hit is not None:
        basis_ref, sp_ref, ctx = hit
        if basis_ref() is basis and sp_ref() is shell_pairs:
            return ctx
        _cuda_ctx_cache.pop(key, None)

    # Simple LRU eviction (best-effort).
    if len(_cuda_ctx_cache) >= int(_CUDA_CTX_CACHE_MAX):
        _cuda_ctx_cache.pop(next(iter(_cuda_ctx_cache)), None)

    dbasis = cueri_gpu.to_device_basis_ss(basis)
    dsp = cueri_gpu.to_device_shell_pairs(shell_pairs)
    dpt = cueri_gpu.build_pair_tables_ss_device(dbasis, dsp, threads=int(threads_i), stream=stream)

    ctx = ERI4cDerivContractionCUDAContext(dsp=dsp, dbasis=dbasis, dpt=dpt, device_id=int(dev_id))
    _cuda_ctx_cache[key] = (weakref.ref(basis), weakref.ref(shell_pairs), ctx)
    return ctx


def make_eri4c_deriv_contraction_cpu_context(
    basis: BasisCartSoA, shell_pairs: ShellPairs, pair_tables: PairTablesCPU
) -> ERI4cDerivContractionCPUContext:
    return ERI4cDerivContractionCPUContext(
        shell_cxyz=np.asarray(basis.shell_cxyz, dtype=np.float64, order="C"),
        shell_l=np.asarray(basis.shell_l, dtype=np.int32, order="C"),
        shell_prim_start=np.asarray(basis.shell_prim_start, dtype=np.int32, order="C"),
        shell_nprim=np.asarray(basis.shell_nprim, dtype=np.int32, order="C"),
        prim_exp=np.asarray(basis.prim_exp, dtype=np.float64, order="C"),
        sp_A=np.asarray(shell_pairs.sp_A, dtype=np.int32, order="C"),
        sp_B=np.asarray(shell_pairs.sp_B, dtype=np.int32, order="C"),
        sp_pair_start=np.asarray(shell_pairs.sp_pair_start, dtype=np.int32, order="C"),
        sp_npair=np.asarray(shell_pairs.sp_npair, dtype=np.int32, order="C"),
        pair_eta=np.asarray(pair_tables.pair_eta, dtype=np.float64, order="C"),
        pair_Px=np.asarray(pair_tables.pair_Px, dtype=np.float64, order="C"),
        pair_Py=np.asarray(pair_tables.pair_Py, dtype=np.float64, order="C"),
        pair_Pz=np.asarray(pair_tables.pair_Pz, dtype=np.float64, order="C"),
        pair_cK=np.asarray(pair_tables.pair_cK, dtype=np.float64, order="C"),
    )


def eri4c_deriv_contracted_cart_class_tasks_cpu(
    ctx: ERI4cDerivContractionCPUContext,
    *,
    task_spAB: np.ndarray,
    task_spCD: np.ndarray,
    bar_eri: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    """Contract analytic 4c ERI nuclear derivatives for a (la,lb,lc,ld)-homogeneous task list (CPU).

    Parameters
    - ctx: pre-packed CPU context (basis + shell-pairs + pair-tables)
    - task_spAB/task_spCD: int32 arrays (nt,)
    - bar_eri: float64 array (nt, nAB, nCD) in cuERI tile layout
    - threads: OpenMP threads inside the CPU extension (0 => serial)

    Returns
    - out: float64 array (nt, 4, 3) with centers [A,B,C,D] and xyz.
    """

    task_spAB = np.asarray(task_spAB, dtype=np.int32).ravel()
    task_spCD = np.asarray(task_spCD, dtype=np.int32).ravel()
    if task_spAB.shape != task_spCD.shape:
        raise ValueError("task_spAB/task_spCD shape mismatch")
    nt = int(task_spAB.shape[0])
    if nt == 0:
        return np.empty((0, 4, 3), dtype=np.float64)

    bar_eri = np.asarray(bar_eri, dtype=np.float64, order="C")
    if bar_eri.ndim != 3 or int(bar_eri.shape[0]) != nt:
        raise ValueError("bar_eri must have shape (nt, nAB, nCD)")

    threads_i = int(threads)
    if threads_i < 0:
        raise ValueError("threads must be >= 0")

    _ext = _require_eri_cpu_ext()
    fn = getattr(_ext, "eri_rys_deriv_contracted_cart_sp_batch_cy", None)
    if fn is None:  # pragma: no cover
        raise RuntimeError("CPU ERI extension is missing eri_rys_deriv_contracted_cart_sp_batch_cy; rebuild it")

    # Group by spAB and dispatch batch kernels: fixed spAB, varying spCD.
    perm64 = np.argsort(task_spAB, kind="stable")
    perm = np.asarray(perm64, dtype=np.int32)
    spab_sorted = task_spAB[perm]

    out = np.empty((nt, 4, 3), dtype=np.float64)
    changes = np.nonzero(spab_sorted[1:] != spab_sorted[:-1])[0] + 1
    bounds = np.concatenate(([0], changes, [nt])).astype(np.int32, copy=False)

    for i0, i1 in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
        i0_i = int(i0)
        i1_i = int(i1)
        idx = perm[i0_i:i1_i]
        spAB = int(spab_sorted[i0_i])
        spCD = np.asarray(task_spCD[idx], dtype=np.int32, order="C")
        bar = np.asarray(bar_eri[idx], dtype=np.float64, order="C")
        out[idx] = fn(
            ctx.shell_cxyz,
            ctx.shell_l,
            ctx.shell_prim_start,
            ctx.shell_nprim,
            ctx.prim_exp,
            ctx.sp_A,
            ctx.sp_B,
            ctx.sp_pair_start,
            ctx.sp_npair,
            ctx.pair_eta,
            ctx.pair_Px,
            ctx.pair_Py,
            ctx.pair_Pz,
            ctx.pair_cK,
            int(spAB),
            spCD,
            bar,
            threads_i,
        )

    return out


def eri4c_deriv_contracted_cart_class_tasks_cuda(
    *,
    task_spAB,
    task_spCD,
    dsp,
    dbasis,
    dpt,
    la: int,
    lb: int,
    lc: int,
    ld: int,
    bar_eri,
    threads: int = 256,
    sync: bool = True,
):
    """Contract analytic 4c ERI nuclear derivatives for a (la,lb,lc,ld)-homogeneous task list (CUDA)."""

    try:
        import cupy as cp  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CUDA contraction requires CuPy") from e

    from asuka.cueri import gpu as cueri_gpu  # noqa: PLC0415

    task_spAB_dev = cp.asarray(task_spAB, dtype=cp.int32)
    task_spCD_dev = cp.asarray(task_spCD, dtype=cp.int32)
    bar_dev = cp.asarray(bar_eri, dtype=cp.float64)
    return cueri_gpu.eri_rys_generic_deriv_contracted_device(
        task_spAB_dev,
        task_spCD_dev,
        dsp,
        dbasis,
        dpt,
        int(la),
        int(lb),
        int(lc),
        int(ld),
        bar_dev,
        threads=int(threads),
        sync=bool(sync),
    )


def accumulate_eri4c_task_derivs_to_atoms(
    *,
    task_spAB: np.ndarray,
    task_spCD: np.ndarray,
    shell_pairs: ShellPairs,
    shell_to_atom: np.ndarray,
    task_derivs: np.ndarray,
    natm: int | None = None,
) -> np.ndarray:
    """Accumulate per-task (A,B,C,D) derivative blocks into an (natm,3) array."""

    task_spAB = np.asarray(task_spAB, dtype=np.int32).ravel()
    task_spCD = np.asarray(task_spCD, dtype=np.int32).ravel()
    if task_spAB.shape != task_spCD.shape:
        raise ValueError("task_spAB/task_spCD shape mismatch")
    nt = int(task_spAB.shape[0])

    task_derivs = np.asarray(task_derivs, dtype=np.float64)
    if task_derivs.shape != (nt, 4, 3):
        raise ValueError("task_derivs must have shape (nt, 4, 3)")

    shell_to_atom = np.asarray(shell_to_atom, dtype=np.int32).ravel()
    if natm is None:
        natm_i = int(shell_to_atom.max()) + 1 if int(shell_to_atom.size) else 0
    else:
        natm_i = int(natm)
    if natm_i < 0:
        raise ValueError("natm must be >= 0")

    grad = np.zeros((natm_i, 3), dtype=np.float64)

    sp_A = np.asarray(shell_pairs.sp_A, dtype=np.int32).ravel()
    sp_B = np.asarray(shell_pairs.sp_B, dtype=np.int32).ravel()

    shellA = sp_A[task_spAB]
    shellB = sp_B[task_spAB]
    shellC = sp_A[task_spCD]
    shellD = sp_B[task_spCD]

    atomA = shell_to_atom[shellA]
    atomB = shell_to_atom[shellB]
    atomC = shell_to_atom[shellC]
    atomD = shell_to_atom[shellD]

    np.add.at(grad, atomA, task_derivs[:, 0, :])
    np.add.at(grad, atomB, task_derivs[:, 1, :])
    np.add.at(grad, atomC, task_derivs[:, 2, :])
    np.add.at(grad, atomD, task_derivs[:, 3, :])
    return grad


__all__ = [
    "ERI4cDerivContractionCUDAContext",
    "ERI4cDerivContractionCPUContext",
    "accumulate_eri4c_task_derivs_to_atoms",
    "eri4c_deriv_contracted_cart_class_tasks_cpu",
    "eri4c_deriv_contracted_cart_class_tasks_cuda",
    "make_eri4c_deriv_contraction_cuda_context",
    "make_eri4c_deriv_contraction_cpu_context",
]
