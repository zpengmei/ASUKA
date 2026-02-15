from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import weakref

from .auxbasis import infer_jkfit_auxbasis
from .cart import ncart
from .stream import stream_ctx, stream_ptr


_DF_METRIC_CHOL_CACHE_MAX = 4
_df_metric_chol_cache: dict[tuple, tuple[weakref.ref, object]] = {}

_DF_STREAM_RYS_PLAN_CACHE_MAX = 2
_df_stream_rys_plan_cache: dict[tuple, tuple[weakref.ref, weakref.ref, object]] = {}

_DF_STREAM_XBLOCK_PLAN_CACHE_MAX = 2
_df_stream_xblock_plan_cache: dict[tuple, tuple[weakref.ref, weakref.ref, object]] = {}


@dataclass(frozen=True)
class _DFStreamRysPlan:
    dbasis: object
    dsp: object
    pt: object
    sp_all: object
    shell_l: np.ndarray
    nsp_ao: int


@dataclass(frozen=True)
class _DFStreamXBlockBatchPlan:
    kernel_tasks: object
    kernel_class_id: np.int32
    transpose: bool
    a0_dev: object
    b0_dev: object
    p0_dev: object
    nB: int
    nAB: int
    nP: int
    ntasks: int
    native: bool


@dataclass(frozen=True)
class _DFStreamXBlockPlanBlock:
    shell_start: int
    shell_stop: int
    p0: int
    p1: int
    batches: list[_DFStreamXBlockBatchPlan]


@dataclass(frozen=True)
class _DFStreamXBlockPlan:
    aux_block_naux: int
    blocks: list[_DFStreamXBlockPlanBlock]


def _trim_df_cache(cache: dict, max_items: int = _DF_METRIC_CHOL_CACHE_MAX) -> None:
    while len(cache) > int(max_items):
        cache.pop(next(iter(cache)))


def _get_cached_metric_cholesky(aux_basis, *, backend: str, stream=None, profile: dict | None = None):
    """Cache and return the aux metric Cholesky factor L for a packed aux basis."""

    import cupy as cp

    dev = int(cp.cuda.runtime.getDevice())
    key = (dev, id(aux_basis), str(backend).lower().strip())
    hit = _df_metric_chol_cache.get(key)
    if hit is not None:
        aux_ref, L = hit
        if aux_ref() is aux_basis:
            if profile is not None:
                prof = profile.setdefault("metric_cholesky", {})
                prof["cache_hit"] = True
                prof.setdefault("ms", 0.0)
            return L
        del _df_metric_chol_cache[key]

    # Compute on the requested stream (if provided) and synchronize before caching
    # so that subsequent use on any stream is safe.
    if profile is not None:
        prof = profile.setdefault("metric_cholesky", {})
        prof["cache_hit"] = False
        prof["backend"] = str(backend)
        try:
            shell_ao_start = np.asarray(getattr(aux_basis, "shell_ao_start"), dtype=np.int64).ravel()
            shell_l = np.asarray(getattr(aux_basis, "shell_l"), dtype=np.int64).ravel()
            nfunc = np.asarray([ncart(int(l)) for l in shell_l], dtype=np.int64)
            prof["naux"] = int(np.max(shell_ao_start + nfunc)) if shell_l.size else 0
        except Exception:
            prof["naux"] = None
    start = cp.cuda.Event() if profile is not None else None
    end = cp.cuda.Event() if profile is not None else None
    with stream_ctx(stream):
        s = cp.cuda.get_current_stream()
        if start is not None and end is not None:
            start.record(s)
        V = metric_2c2e_basis(aux_basis, stream=stream, backend=backend)
        L = cholesky_metric(V)
        if start is not None and end is not None:
            end.record(s)
            end.synchronize()
            if profile is not None:
                prof["ms"] = float(cp.cuda.get_elapsed_time(start, end))

        # Synchronize before caching so later use on any stream is safe.
        s.synchronize()

    _df_metric_chol_cache[key] = (weakref.ref(aux_basis), L)
    _trim_df_cache(_df_metric_chol_cache)
    return L


def _get_cached_df_stream_rys_plan(ao_basis, aux_basis, *, stream=None, threads: int, profile: dict | None = None) -> _DFStreamRysPlan:
    """Cache and return the precomputed device tables for streamed DF (general-l Rys backend)."""

    import cupy as cp

    dev = int(cp.cuda.runtime.getDevice())
    key = (dev, id(ao_basis), id(aux_basis))
    hit = _df_stream_rys_plan_cache.get(key)
    if hit is not None:
        ao_ref, aux_ref, plan = hit
        if ao_ref() is ao_basis and aux_ref() is aux_basis:
            if profile is not None:
                prof = profile.setdefault("yt_stream", {})
                prof["plan_cache_hit"] = True
                prof.setdefault("plan_build_ms", 0.0)
            return plan
        del _df_stream_rys_plan_cache[key]

    if profile is not None:
        prof = profile.setdefault("yt_stream", {})
        prof["plan_cache_hit"] = False

    from types import SimpleNamespace

    from .shell_pairs import ShellPairs, build_shell_pairs_l_order
    from .gpu import build_pair_tables_ss_device, to_device_basis_ss, to_device_shell_pairs

    # ---- Build combined basis arrays (AO + AUX + dummy) and device tables once ----
    n_shell_ao = int(ao_basis.shell_cxyz.shape[0])
    n_shell_aux = int(aux_basis.shell_cxyz.shape[0])
    dummy_shell = int(n_shell_ao + n_shell_aux)

    ao_prim_n = int(np.asarray(ao_basis.prim_exp, dtype=np.float64).shape[0])
    aux_prim_n = int(np.asarray(aux_basis.prim_exp, dtype=np.float64).shape[0])

    shell_cxyz = np.concatenate(
        [
            np.asarray(ao_basis.shell_cxyz, dtype=np.float64),
            np.asarray(aux_basis.shell_cxyz, dtype=np.float64),
            np.zeros((1, 3), dtype=np.float64),
        ],
        axis=0,
    )
    shell_prim_start = np.concatenate(
        [
            np.asarray(ao_basis.shell_prim_start, dtype=np.int32),
            np.asarray(aux_basis.shell_prim_start, dtype=np.int32) + np.int32(ao_prim_n),
            np.asarray([ao_prim_n + aux_prim_n], dtype=np.int32),
        ]
    )
    shell_nprim = np.concatenate(
        [
            np.asarray(ao_basis.shell_nprim, dtype=np.int32),
            np.asarray(aux_basis.shell_nprim, dtype=np.int32),
            np.asarray([1], dtype=np.int32),
        ]
    )
    prim_exp = np.concatenate(
        [np.asarray(ao_basis.prim_exp, dtype=np.float64), np.asarray(aux_basis.prim_exp, dtype=np.float64), np.asarray([0.0])]
    )
    prim_coef = np.concatenate(
        [np.asarray(ao_basis.prim_coef, dtype=np.float64), np.asarray(aux_basis.prim_coef, dtype=np.float64), np.asarray([1.0])]
    )

    ao_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    aux_l = np.asarray(aux_basis.shell_l, dtype=np.int32).ravel()
    shell_l = np.concatenate([ao_l, aux_l, np.asarray([0], dtype=np.int32)])

    basis_like = SimpleNamespace(
        shell_cxyz=shell_cxyz,
        shell_prim_start=shell_prim_start,
        shell_nprim=shell_nprim,
        prim_exp=prim_exp,
        prim_coef=prim_coef,
    )

    # ShellPairs:
    # - AO side: all unique unordered AO shell pairs, oriented with l(A) >= l(B).
    sp_ao = build_shell_pairs_l_order(ao_basis)
    nsp_ao = int(sp_ao.sp_A.shape[0])

    # - Aux side: (P, dummy) for each aux shell P.
    aux_shell_idx = np.arange(n_shell_aux, dtype=np.int32)
    sp_aux_A = (aux_shell_idx + np.int32(n_shell_ao)).astype(np.int32, copy=False)
    sp_aux_B = np.full((int(aux_shell_idx.size),), int(dummy_shell), dtype=np.int32)
    sp_aux_npair = np.asarray(aux_basis.shell_nprim, dtype=np.int32).ravel().astype(np.int32, copy=False)

    sp_A = np.concatenate([np.asarray(sp_ao.sp_A, dtype=np.int32), sp_aux_A])
    sp_B = np.concatenate([np.asarray(sp_ao.sp_B, dtype=np.int32), sp_aux_B])
    sp_npair = np.concatenate([np.asarray(sp_ao.sp_npair, dtype=np.int32), sp_aux_npair])
    sp_pair_start = np.empty((int(sp_npair.shape[0]) + 1,), dtype=np.int32)
    sp_pair_start[0] = 0
    sp_pair_start[1:] = np.cumsum(sp_npair, dtype=np.int32)
    sp_all = ShellPairs(sp_A=sp_A, sp_B=sp_B, sp_npair=sp_npair, sp_pair_start=sp_pair_start)

    t0 = cp.cuda.Event() if profile is not None else None
    t1 = cp.cuda.Event() if profile is not None else None
    with stream_ctx(stream):
        s0 = cp.cuda.get_current_stream()
        if t0 is not None and t1 is not None:
            t0.record(s0)

        dbasis = to_device_basis_ss(basis_like)
        dsp = to_device_shell_pairs(sp_all)
        pt = build_pair_tables_ss_device(dbasis, dsp, stream=stream, threads=threads)

        if t0 is not None and t1 is not None:
            t1.record(s0)
            t1.synchronize()
            if profile is not None:
                prof = profile.setdefault("yt_stream", {})
                prof["plan_build_ms"] = float(cp.cuda.get_elapsed_time(t0, t1))

        s0.synchronize()

    plan = _DFStreamRysPlan(dbasis=dbasis, dsp=dsp, pt=pt, sp_all=sp_all, shell_l=shell_l, nsp_ao=nsp_ao)
    _df_stream_rys_plan_cache[key] = (weakref.ref(ao_basis), weakref.ref(aux_basis), plan)
    _trim_df_cache(_df_stream_rys_plan_cache, max_items=_DF_STREAM_RYS_PLAN_CACHE_MAX)
    return plan


def _get_cached_df_stream_xblock_plan(
    ao_basis,
    aux_basis,
    *,
    aux_block_naux: int,
    base_plan: _DFStreamRysPlan,
    stream=None,
    profile: dict | None = None,
) -> _DFStreamXBlockPlan:
    """Cache per-aux-block kernel batches + scatter indices for the streamed X-block path.

    This avoids re-planning kernel batches (and re-building index arrays) on every streamed
    active build, which becomes a dominant overhead for larger systems.
    """

    import cupy as cp

    aux_block_naux = int(aux_block_naux)
    if aux_block_naux <= 0:
        raise ValueError("aux_block_naux must be > 0")

    dev = int(cp.cuda.runtime.getDevice())
    key = (dev, id(ao_basis), id(aux_basis), int(aux_block_naux))
    hit = _df_stream_xblock_plan_cache.get(key)
    if hit is not None:
        ao_ref, aux_ref, plan = hit
        if ao_ref() is ao_basis and aux_ref() is aux_basis:
            if profile is not None:
                prof = profile.setdefault("yt_stream", {})
                prof["xblock_plan_cache_hit"] = True
                prof.setdefault("xblock_plan_build_ms", 0.0)
            return plan
        del _df_stream_xblock_plan_cache[key]

    if profile is not None:
        prof = profile.setdefault("yt_stream", {})
        prof["xblock_plan_cache_hit"] = False

    from .eri_dispatch import plan_kernel_batches_spd
    from .tasks import TaskList, decode_eri_class_id, eri_class_id

    # Native kernel set (same as run_kernel_batch_spd dispatch).
    native_set = {
        int(eri_class_id(0, 0, 0, 0)),
        int(eri_class_id(1, 0, 0, 0)),
        int(eri_class_id(1, 1, 0, 0)),
        int(eri_class_id(1, 0, 1, 0)),
        int(eri_class_id(1, 1, 1, 0)),
        int(eri_class_id(1, 1, 1, 1)),
        int(eri_class_id(2, 0, 0, 0)),
        int(eri_class_id(2, 2, 0, 0)),
        int(eri_class_id(0, 0, 2, 1)),
        int(eri_class_id(1, 0, 2, 0)),
        int(eri_class_id(1, 0, 2, 1)),
        int(eri_class_id(1, 0, 2, 2)),
        int(eri_class_id(1, 1, 2, 0)),
        int(eri_class_id(1, 1, 2, 1)),
        int(eri_class_id(1, 1, 2, 2)),
        int(eri_class_id(2, 0, 2, 0)),
        int(eri_class_id(2, 0, 2, 1)),
        int(eri_class_id(2, 0, 2, 2)),
        int(eri_class_id(3, 1, 0, 0)),
        int(eri_class_id(3, 2, 0, 0)),
        int(eri_class_id(3, 3, 0, 0)),
        int(eri_class_id(3, 1, 1, 0)),
        int(eri_class_id(3, 2, 1, 0)),
        int(eri_class_id(3, 3, 1, 0)),
        int(eri_class_id(3, 1, 2, 0)),
        int(eri_class_id(3, 2, 2, 0)),
        int(eri_class_id(3, 3, 2, 0)),
        int(eri_class_id(0, 0, 3, 0)),
        int(eri_class_id(1, 0, 3, 0)),
        int(eri_class_id(1, 1, 3, 0)),
        int(eri_class_id(2, 0, 3, 0)),
        int(eri_class_id(3, 0, 3, 0)),
        int(eri_class_id(2, 1, 3, 0)),
        int(eri_class_id(3, 1, 3, 0)),
        int(eri_class_id(2, 2, 3, 0)),
        int(eri_class_id(3, 2, 3, 0)),
        int(eri_class_id(3, 3, 3, 0)),
        int(eri_class_id(0, 0, 4, 0)),
        int(eri_class_id(1, 0, 4, 0)),
        int(eri_class_id(1, 1, 4, 0)),
        int(eri_class_id(2, 0, 4, 0)),
        int(eri_class_id(3, 0, 4, 0)),
        int(eri_class_id(2, 1, 4, 0)),
        int(eri_class_id(3, 1, 4, 0)),
        int(eri_class_id(2, 2, 4, 0)),
        int(eri_class_id(3, 2, 4, 0)),
        int(eri_class_id(3, 3, 4, 0)),
        int(eri_class_id(2, 1, 2, 1)),
        int(eri_class_id(2, 1, 2, 2)),
        int(eri_class_id(2, 2, 2, 2)),
    }

    sp_all = base_plan.sp_all
    shell_l = base_plan.shell_l
    nsp_ao = int(base_plan.nsp_ao)

    # Precompute AO shell-pair indices (constant across aux blocks).
    ab_idx = np.arange(nsp_ao, dtype=np.int32)

    aux_blocks = plan_aux_blocks_cart(aux_basis, max_block_naux=aux_block_naux)
    blocks: list[_DFStreamXBlockPlanBlock] = []

    t0 = cp.cuda.Event() if profile is not None else None
    t1 = cp.cuda.Event() if profile is not None else None
    with stream_ctx(stream):
        s0 = cp.cuda.get_current_stream()
        if t0 is not None and t1 is not None:
            t0.record(s0)

        for shell_start, shell_stop, p0_block, p1_block in aux_blocks:
            shell_start = int(shell_start)
            shell_stop = int(shell_stop)
            p0_block = int(p0_block)
            p1_block = int(p1_block)
            if shell_stop <= shell_start:
                continue

            p_shells = np.arange(shell_start, shell_stop, dtype=np.int32)
            n_aux_shell_blk = int(p_shells.size)
            if n_aux_shell_blk == 0:
                continue

            # Tasks: all (AO shell pair, aux shell) combinations for this aux-shell block.
            task_ab = np.repeat(ab_idx, n_aux_shell_blk)
            task_cd = (nsp_ao + np.tile(p_shells, int(ab_idx.size))).astype(np.int32, copy=False)
            tasks = TaskList(task_spAB=task_ab, task_spCD=task_cd)

            batches = plan_kernel_batches_spd(tasks, shell_pairs=sp_all, shell_l=shell_l)
            batch_plans: list[_DFStreamXBlockBatchPlan] = []

            for batch in batches:
                idx_full = np.asarray(batch.task_idx, dtype=np.int32).ravel()
                if idx_full.size == 0:
                    continue

                spab_full = np.asarray(tasks.task_spAB[idx_full], dtype=np.int32).ravel()
                spcd_full = np.asarray(tasks.task_spCD[idx_full], dtype=np.int32).ravel()

                A_full = np.asarray(sp_all.sp_A[spab_full], dtype=np.int32).ravel()
                B_full = np.asarray(sp_all.sp_B[spab_full], dtype=np.int32).ravel()
                P_shell_full = (spcd_full - np.int32(nsp_ao)).astype(np.int32, copy=False)

                a0_full = np.asarray(ao_basis.shell_ao_start[A_full], dtype=np.int32).ravel()
                b0_full = np.asarray(ao_basis.shell_ao_start[B_full], dtype=np.int32).ravel()
                p0_full = (np.asarray(aux_basis.shell_ao_start[P_shell_full], dtype=np.int32).ravel() - np.int32(p0_block)).astype(
                    np.int32, copy=False
                )

                a0_dev = cp.ascontiguousarray(cp.asarray(a0_full, dtype=cp.int32))
                b0_dev = cp.ascontiguousarray(cp.asarray(b0_full, dtype=cp.int32))
                p0_dev = cp.ascontiguousarray(cp.asarray(p0_full, dtype=cp.int32))

                la, lb, lc, ld = decode_eri_class_id(int(batch.kernel_class_id))
                if bool(batch.transpose):
                    la, lb, lc, ld = lc, ld, la, lb
                nAB = int(ncart(int(la))) * int(ncart(int(lb)))
                nP = int(ncart(int(lc))) * int(ncart(int(ld)))
                nB = int(ncart(int(lb)))

                # Store kernel-task lists as contiguous int32 arrays.
                kt_ab = np.asarray(batch.kernel_tasks.task_spAB, dtype=np.int32).ravel()
                kt_cd = np.asarray(batch.kernel_tasks.task_spCD, dtype=np.int32).ravel()
                kernel_tasks = TaskList(task_spAB=kt_ab, task_spCD=kt_cd)

                batch_plans.append(
                    _DFStreamXBlockBatchPlan(
                        kernel_tasks=kernel_tasks,
                        kernel_class_id=np.int32(batch.kernel_class_id),
                        transpose=bool(batch.transpose),
                        a0_dev=a0_dev,
                        b0_dev=b0_dev,
                        p0_dev=p0_dev,
                        nB=int(nB),
                        nAB=int(nAB),
                        nP=int(nP),
                        ntasks=int(idx_full.size),
                        native=bool(int(batch.kernel_class_id) in native_set),
                    )
                )

            blocks.append(
                _DFStreamXBlockPlanBlock(
                    shell_start=int(shell_start),
                    shell_stop=int(shell_stop),
                    p0=int(p0_block),
                    p1=int(p1_block),
                    batches=batch_plans,
                )
            )

        if t0 is not None and t1 is not None:
            t1.record(s0)
            t1.synchronize()
            if profile is not None:
                prof = profile.setdefault("yt_stream", {})
                prof["xblock_plan_build_ms"] = float(cp.cuda.get_elapsed_time(t0, t1))

        s0.synchronize()

    plan = _DFStreamXBlockPlan(aux_block_naux=int(aux_block_naux), blocks=blocks)
    _df_stream_xblock_plan_cache[key] = (weakref.ref(ao_basis), weakref.ref(aux_basis), plan)
    _trim_df_cache(_df_stream_xblock_plan_cache, max_items=_DF_STREAM_XBLOCK_PLAN_CACHE_MAX)
    return plan


def recommended_auxbasis(mol, *, xc: str = "HF", mp2fit: bool = False):
    """Return a "corresponding" DF auxiliary basis for the given orbital basis.

    This function is PySCF-independent and currently implements a small set of
    heuristics for selecting JKFIT auxiliary bases from common orbital basis
    names.

    Notes
    - Currently, only JKFIT-style recommendations are implemented (MP2FIT is
      ignored). For production, pass `auxbasis` explicitly if you need a specific
      RI family.
    """

    # Signature keeps `xc/mp2fit` for forwards compatibility, but the current
    # in-house selector does not use them.
    _ = str(xc)
    _ = bool(mp2fit)

    # Try to extract a minimal element list from `mol` without importing PySCF.
    elements: list[str] = []
    natm = getattr(mol, "natm", None)
    if natm is not None:
        try:
            natm_i = int(natm)
            atom_symbol = getattr(mol, "atom_symbol", None)
            if callable(atom_symbol):
                elements = [str(atom_symbol(i)) for i in range(natm_i)]
        except Exception:
            elements = []

    if not elements:
        # Best-effort fallback: parse from `mol.atom` string/list.
        atom = getattr(mol, "atom", None)
        if isinstance(atom, str):
            for frag in atom.split(";"):
                tok = frag.strip().split()
                if tok:
                    elements.append(tok[0])
        elif isinstance(atom, (list, tuple)):
            for item in atom:
                if isinstance(item, str):
                    tok = item.strip().split()
                    if tok:
                        elements.append(tok[0])
                elif isinstance(item, (list, tuple)) and item:
                    elements.append(str(item[0]))

    elements = sorted(set(elements))
    if not elements:
        raise ValueError("could not determine element symbols from mol; pass auxbasis explicitly")

    orbital_basis = getattr(mol, "basis", None)
    if orbital_basis is None:
        raise ValueError("mol does not provide a 'basis' attribute; pass auxbasis explicitly")

    return infer_jkfit_auxbasis(orbital_basis, elements)


def metric_2c2e_basis(aux_basis, *, stream=None, backend: str = "gpu_rys", mode: str = "warp", threads: int = 256):
    """Compute the aux Coulomb metric V(P,Q) = (P|Q) from packed basis objects (GPU).

    This is a lower-level entrypoint that avoids any PySCF objects. It is
    intended for the eventual "no-PySCF runtime" path where AO/AUX bases are
    packed by cuERI itself.
    """

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for DF metric_2c2e_basis") from e

    backend = str(backend).lower().strip()
    if backend not in ("gpu_ss", "gpu_sp", "gpu_rys"):
        raise ValueError("backend must be one of: 'gpu_ss', 'gpu_sp', 'gpu_rys'")

    if backend == "gpu_ss":
        from .gpu import df_metric_2c2e_ss_device, has_cuda_ext

        if not has_cuda_ext():
            raise RuntimeError("cuERI CUDA extension not available; build via `python -m asuka.cueri.build_cuda_ext`")
        with stream_ctx(stream):
            return cp.ascontiguousarray(df_metric_2c2e_ss_device(aux_basis, stream=stream, threads=int(threads)))

    if backend == "gpu_sp":
        from .gpu import df_metric_2c2e_sp_device, has_cuda_ext

        if not has_cuda_ext():
            raise RuntimeError("cuERI CUDA extension not available; build via `python -m asuka.cueri.build_cuda_ext`")
        with stream_ctx(stream):
            return cp.ascontiguousarray(
                df_metric_2c2e_sp_device(
                    aux_basis,
                    stream=stream,
                    threads=int(threads),
                    mode=str(mode),
                )
            )

    from .gpu import df_metric_2c2e_rys_device, has_cuda_ext

    if not has_cuda_ext():
        raise RuntimeError("cuERI CUDA extension not available; build via `python -m asuka.cueri.build_cuda_ext`")
    with stream_ctx(stream):
        return cp.ascontiguousarray(
            df_metric_2c2e_rys_device(
                aux_basis,
                stream=stream,
                threads=int(threads),
                mode=str(mode),
            )
        )


def int3c2e_basis(
    ao_basis,
    aux_basis,
    *,
    stream=None,
    backend: str = "gpu_rys",
    mode: str = "warp",
    threads: int = 256,
    ao_contract_mode: str = "auto",
    profile: dict | None = None,
):
    """Compute 3-center Coulomb integrals X(μ,ν,P) = (μν|P) from packed bases (GPU)."""

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for DF int3c2e_basis") from e

    backend = str(backend).lower().strip()
    if backend not in ("gpu_ss", "gpu_sp", "gpu_rys"):
        raise ValueError("backend must be one of: 'gpu_ss', 'gpu_sp', 'gpu_rys'")

    if backend == "gpu_ss":
        from .gpu import df_int3c2e_ss_device, has_cuda_ext

        if not has_cuda_ext():
            raise RuntimeError("cuERI CUDA extension not available; build via `python -m asuka.cueri.build_cuda_ext`")
        with stream_ctx(stream):
            return cp.ascontiguousarray(df_int3c2e_ss_device(ao_basis, aux_basis, stream=stream, threads=int(threads)))

    if backend == "gpu_sp":
        from .gpu import df_int3c2e_sp_device, has_cuda_ext

        if not has_cuda_ext():
            raise RuntimeError("cuERI CUDA extension not available; build via `python -m asuka.cueri.build_cuda_ext`")
        with stream_ctx(stream):
            return cp.ascontiguousarray(
                df_int3c2e_sp_device(
                    ao_basis,
                    aux_basis,
                    stream=stream,
                    threads=int(threads),
                    mode=str(mode),
                    ao_contract_mode=str(ao_contract_mode),
                    profile=profile,
                )
            )

    from .gpu import df_int3c2e_rys_device, has_cuda_ext

    if not has_cuda_ext():
        raise RuntimeError("cuERI CUDA extension not available; build via `python -m asuka.cueri.build_cuda_ext`")
    with stream_ctx(stream):
        return cp.ascontiguousarray(
            df_int3c2e_rys_device(
                ao_basis,
                aux_basis,
                stream=stream,
                threads=int(threads),
                mode=str(mode),
                ao_contract_mode=str(ao_contract_mode),
                profile=profile,
            )
        )


def cholesky_metric(V):
    """Cholesky factorization V = L L^T on GPU (returns lower-triangular L)."""

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for DF cholesky_metric") from e

    V = cp.asarray(V, dtype=cp.float64)
    if V.ndim != 2 or V.shape[0] != V.shape[1]:
        raise ValueError("V must be a square 2D array")
    return cp.linalg.cholesky(V)


def whiten_3c2e(X, L):
    """Whiten 3c2e integrals using metric Cholesky: B = X @ L^{-T}.

    Inputs
    - X: (nao, nao, naux) (CuPy)
    - L: (naux, naux) lower-triangular Cholesky factor (CuPy)

    Returns
    - B: (nao, nao, naux) (CuPy)
    """

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for DF whiten_3c2e") from e

    import cupyx.scipy.linalg as cpx_linalg

    X = cp.asarray(X, dtype=cp.float64)
    L = cp.asarray(L, dtype=cp.float64)
    if X.ndim != 3:
        raise ValueError("X must have shape (nao, nao, naux)")
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square 2D array")
    nao0, nao1, naux = map(int, X.shape)
    if nao0 != nao1:
        raise ValueError("X must have shape (nao, nao, naux)")
    if int(L.shape[0]) != naux:
        raise ValueError(f"L has shape {L.shape}, but X implies naux={naux}")

    X_flat = X.reshape((nao0 * nao0, naux))
    # B^T = L^{-1} X^T  (L is lower)
    BT = cpx_linalg.solve_triangular(L, X_flat.T, lower=True, trans="N", unit_diagonal=False, overwrite_b=False)
    # Normalize layout to C-contiguous so downstream (J/K, transforms) have
    # stable performance regardless of the input X memory order/strides.
    B_flat = cp.ascontiguousarray(BT.T)
    B = B_flat.reshape((nao0, nao0, naux))
    return cp.ascontiguousarray(B)


def active_Lfull_from_B(B, C_active):
    """Transform whitened 3c2e factors to active MO DF vectors.

    Computes:
      L_{pq}^Q = Σ_{μν} C_{μp} C_{νq} B_{μν}^Q

    Returns ordered-pair layout:
      L_full[p*norb + q, Q]
    """

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for DF active_Lfull_from_B") from e

    B = cp.asarray(B, dtype=cp.float64)
    C_active = cp.asarray(C_active, dtype=cp.float64)
    C_active = cp.ascontiguousarray(C_active)

    if B.ndim != 3:
        raise ValueError("B must have shape (nao, nao, naux)")
    nao0, nao1, naux = map(int, B.shape)
    if nao0 != nao1:
        raise ValueError("B must have shape (nao, nao, naux)")
    if C_active.ndim != 2:
        raise ValueError("C_active must have shape (nao, norb)")
    nao, norb = map(int, C_active.shape)
    if nao != nao0:
        raise ValueError(f"C_active has nao={nao}, but B implies nao={nao0}")

    # tmp[p, ν, Q] = Σ_μ C_{μp} B_{μν}^Q
    tmp = C_active.T @ B.reshape((nao, nao * naux))
    tmp = tmp.reshape((norb, nao, naux))

    # out[p, Q, q] = Σ_ν tmp[p, ν, Q] C_{νq}
    out = cp.matmul(tmp.transpose(0, 2, 1), C_active)  # (norb, naux, norb)
    L_pqQ = out.transpose(0, 2, 1)  # (norb, norb, naux)
    return L_pqQ.reshape((norb * norb, naux))


def active_Lfull_from_cached_B_whitened(B_whitened, C_active):
    """Transform a cached whitened AO DF tensor to active-space DF vectors.

    This is a thin wrapper over :func:`active_Lfull_from_B` to make the cached
    tensor path explicit at call sites.
    """

    return active_Lfull_from_B(B_whitened, C_active)


def _active_YT_streamed_rys_basis(
    ao_basis,
    aux_basis,
    C_active,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "auto",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 4,
    aux_block_naux: int = 256,
    max_tile_bytes: int = 256 * 1024 * 1024,
    strategy: str = "auto",
    ao_contract_mode: str = "auto",
    graph_capture: bool = False,
    profile: dict | None = None,
):
    """Build Y^T[P, pq] = Σ_{μν} C_{μp} C_{νq} X_{μν,P} without materializing X or B.

    This is the core streaming DF path described in `docs/impl_df_stream_active.md`.

    Notes
    - Works with packed Cartesian basis objects (AO + AUX) and evaluates 3c2e tiles on GPU.
    - Returns `YT` with shape (naux, norb*norb) on GPU (CuPy).
    """

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for streamed DF active build") from e

    from .eri_utils import build_pair_coeff_ordered
    from .tasks import TaskList, decode_eri_class_id
    from .basis_utils import shell_nfunc_cart

    from .gpu import CUDA_MAX_L, has_cuda_ext
    from .eri_dispatch import KernelBatch, plan_kernel_batches_spd, run_kernel_batch_spd

    def _launch_cuda_graph(graph_obj, stream_obj):
        # CuPy API compatibility across versions.
        try:
            graph_obj.launch(stream_obj)
        except TypeError:
            graph_obj.launch()

    def _capture_cuda_graph(stream_obj, capture_body):
        # Best-effort capture: if capture is unsupported for this body/runtime,
        # return (False, None, reason) and caller falls back to normal launches.
        try:
            stream_obj.begin_capture()
            capture_body()
            graph_obj = stream_obj.end_capture()
            return True, graph_obj, None
        except Exception as e:  # pragma: no cover - depends on CUDA/CuPy runtime support
            try:
                stream_obj.end_capture()
            except Exception:
                pass
            return False, None, str(e)

    if not has_cuda_ext():
        raise RuntimeError("cuERI CUDA extension not available; build via `python -m asuka.cueri.build_cuda_ext`")

    try:
        from . import _cueri_cuda_ext as _ext
    except Exception as e:  # pragma: no cover
        raise RuntimeError("cuERI CUDA extension not available") from e
    sptr = stream_ptr(stream)

    C_active = cp.asarray(C_active, dtype=cp.float64)
    C_active = cp.ascontiguousarray(C_active)
    if C_active.ndim != 2:
        raise ValueError("C_active must have shape (nao, norb)")
    nao, norb = map(int, C_active.shape)
    if nao <= 0 or norb <= 0:
        raise ValueError("C_active must be non-empty (nao>0 and norb>0)")

    ao_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    aux_l = np.asarray(aux_basis.shell_l, dtype=np.int32).ravel()
    if ao_l.size == 0 or aux_l.size == 0:
        return cp.empty((0, norb * norb), dtype=cp.float64)
    if int(ao_l.max()) > CUDA_MAX_L or int(aux_l.max()) > CUDA_MAX_L:
        raise NotImplementedError(
            f"streamed active DF currently supports only l<={CUDA_MAX_L} for AO and aux shells (cart)"
        )

    # Sanity-check AO dimensions against packed basis AO layout.
    ao_start = np.asarray(ao_basis.shell_ao_start, dtype=np.int32).ravel()
    ao_nfunc = np.asarray(shell_nfunc_cart(ao_basis), dtype=np.int32)
    nao_expected = int(np.max(ao_start + ao_nfunc)) if ao_start.size else 0
    if nao_expected != nao:
        raise ValueError(f"C_active has nao={nao}, but packed ao_basis expects nao={nao_expected}")

    # AUX dimensions.
    aux_start = np.asarray(aux_basis.shell_ao_start, dtype=np.int32).ravel()
    aux_nfunc = np.asarray([ncart(int(l)) for l in aux_l], dtype=np.int32)
    naux = int(np.max(aux_start + aux_nfunc)) if aux_start.size else 0

    nops = int(norb * norb)
    YT = cp.zeros((naux, nops), dtype=cp.float64)

    mode = str(mode).lower().strip()
    if mode not in ("block", "warp", "multiblock", "auto"):
        raise ValueError("mode must be one of: 'block', 'warp', 'multiblock', 'auto'")

    strategy = str(strategy).lower().strip().replace("-", "_")
    if strategy not in ("auto", "x_block", "digest"):
        raise ValueError("strategy must be one of: 'auto', 'x_block', 'digest'")

    ao_contract_mode = str(ao_contract_mode).lower().strip()
    if ao_contract_mode not in ("auto", "expanded", "native_contracted"):
        raise ValueError("ao_contract_mode must be one of: 'auto', 'expanded', 'native_contracted'")
    graph_capture = bool(graph_capture)

    aux_block_naux = int(aux_block_naux)
    if aux_block_naux <= 0:
        raise ValueError("aux_block_naux must be > 0")

    max_tile_bytes = int(max_tile_bytes)
    if max_tile_bytes <= 0:
        raise ValueError("max_tile_bytes must be > 0")

    # Prefer the X-block path whenever possible since it uses large GEMMs and is
    # typically far more GPU-efficient than the digest fallback. If the requested
    # `aux_block_naux` would make X_blk exceed the scratch budget, shrink the
    # aux block size to the largest value that fits X_blk in `max_tile_bytes`.
    # This keeps the execution more steady (avoids long CPU/GPU bubbles).
    max_shell_naux = int(np.max(aux_nfunc)) if aux_nfunc.size else 0
    bytes_per_aux_func_x = int(nao) * int(nao) * 8
    aux_block_naux_user = int(aux_block_naux)
    aux_block_naux_eff = int(aux_block_naux)
    if bytes_per_aux_func_x > 0 and max_shell_naux > 0:
        max_naux_x = int(max_tile_bytes) // int(bytes_per_aux_func_x)
        if max_naux_x >= int(max_shell_naux) and int(max_naux_x) > 0:
            aux_block_naux_eff = min(int(aux_block_naux_eff), int(max_naux_x))
            aux_block_naux_eff = max(int(aux_block_naux_eff), int(max_shell_naux))
            aux_block_naux = int(aux_block_naux_eff)

    # Stream over aux shells in blocks aligned to aux-function boundaries.
    aux_blocks = plan_aux_blocks_cart(aux_basis, max_block_naux=aux_block_naux)
    if not aux_blocks:
        if profile is not None:
            prof = profile.setdefault("yt_stream", {})
            prof["nao"] = int(nao)
            prof["norb"] = int(norb)
            prof["naux"] = int(naux)
            prof["strategy_requested"] = str(strategy)
            prof["ao_contract_mode"] = str(ao_contract_mode)
            prof["graph_capture_requested"] = bool(graph_capture)
            prof.setdefault("graph_capture_used", False)
            prof["nblocks"] = 0
            prof.setdefault("use_x_block", False)
            prof.setdefault("use_digest", False)
            prof.setdefault("kernel_ms", 0.0)
            prof.setdefault("scatter_ms", 0.0)
            prof.setdefault("transform_ms", 0.0)
            prof.setdefault("batches", {})
            prof.setdefault("kernel_dispatch", {})
            if "strategy" not in prof:
                prof["strategy"] = "digest"
        return YT

    # Reuse a single scratch `X_blk` buffer across aux blocks to reduce allocator
    # churn and make the X-block regime more stable.
    #
    # Allocate lazily since some callers may force digest-only execution with a
    # tiny `max_tile_bytes` (in which case no X-block fits).
    xblock_scratch_naux = 0
    bytes_per_aux_func_x = int(nao) * int(nao) * 8
    if bytes_per_aux_func_x > 0:
        for _shell_start, _shell_stop, _p0, _p1 in aux_blocks:
            naux_blk = int(_p1) - int(_p0)
            if naux_blk <= 0:
                continue
            if int(naux_blk) * int(bytes_per_aux_func_x) <= int(max_tile_bytes):
                xblock_scratch_naux = max(int(xblock_scratch_naux), int(naux_blk))
    X_blk_scratch = None

    if profile is not None:
        prof = profile.setdefault("yt_stream", {})
        prof["nao"] = int(nao)
        prof["norb"] = int(norb)
        prof["naux"] = int(naux)
        prof["strategy_requested"] = str(strategy)
        prof["ao_contract_mode"] = str(ao_contract_mode)
        prof["graph_capture_requested"] = bool(graph_capture)
        prof.setdefault("graph_capture_used", False)
        prof["nblocks"] = int(len(aux_blocks))
        prof["aux_block_naux_user"] = int(aux_block_naux_user)
        prof["aux_block_naux_effective"] = int(aux_block_naux_eff)
        prof["xblock_scratch_naux"] = int(xblock_scratch_naux)
        prof.setdefault("use_x_block", False)
        prof.setdefault("use_digest", False)
        prof.setdefault("kernel_ms", 0.0)
        prof.setdefault("scatter_ms", 0.0)
        prof.setdefault("transform_ms", 0.0)
        prof.setdefault("batches", {})
        prof.setdefault("kernel_dispatch", {})

    dispatch_profile = None
    if profile is not None:
        dispatch_profile = profile.setdefault("yt_stream", {}).setdefault("kernel_dispatch", {})

    # Fast path: if the full X(μ,ν,P) tensor fits, reuse the optimized int3c2e plan
    # (including cached device index tables) and do the MO transform as GEMMs.
    if strategy in ("auto", "x_block") and len(aux_blocks) == 1:
        shell_start0, shell_stop0, p0_0, p1_0 = aux_blocks[0]
        if int(shell_start0) == 0 and int(shell_stop0) == int(aux_basis.shell_cxyz.shape[0]) and int(p0_0) == 0 and int(p1_0) == int(naux):
            x_bytes = int(nao) * int(nao) * int(naux) * 8
            if x_bytes <= max_tile_bytes:
                from .gpu import df_int3c2e_rys_device

                start = cp.cuda.Event() if profile is not None else None
                end = cp.cuda.Event() if profile is not None else None
                s0 = cp.cuda.get_current_stream()
                if start is not None and end is not None:
                    start.record(s0)
                X = df_int3c2e_rys_device(
                    ao_basis,
                    aux_basis,
                    stream=stream,
                    threads=threads,
                    mode=mode,
                    work_small_max=work_small_max,
                    work_large_min=work_large_min,
                    blocks_per_task=blocks_per_task,
                    ao_contract_mode=ao_contract_mode,
                )
                if start is not None and end is not None:
                    end.record(s0)
                    end.synchronize()
                    prof = profile.setdefault("yt_stream", {})
                    prof["full_x_int3c2e_ms"] = float(cp.cuda.get_elapsed_time(start, end))

                start2 = cp.cuda.Event() if profile is not None else None
                end2 = cp.cuda.Event() if profile is not None else None
                if start2 is not None and end2 is not None:
                    start2.record(s0)
                L_full = active_Lfull_from_B(X, C_active)  # (nops, naux)
                if start2 is not None and end2 is not None:
                    end2.record(s0)
                    end2.synchronize()
                    prof = profile.setdefault("yt_stream", {})
                    prof["full_x_transform_ms"] = float(cp.cuda.get_elapsed_time(start2, end2))
                    prof["strategy"] = "full_x"
                    prof["use_x_block"] = True
                return cp.ascontiguousarray(L_full.T)

    plan = _get_cached_df_stream_rys_plan(ao_basis, aux_basis, stream=stream, threads=threads, profile=profile)
    dbasis = plan.dbasis
    dsp = plan.dsp
    pt = plan.pt
    sp_all = plan.sp_all
    shell_l = plan.shell_l
    nsp_ao = int(plan.nsp_ao)

    # Cache K_AB^T for each AO shell-pair (spAB) since it depends only on C_active.
    k_ab_t_cache: dict[int, cp.ndarray] = {}

    ab_idx = np.arange(nsp_ao, dtype=np.int32)
    xblock_plan = None
    dummy_task_idx = np.empty((0,), dtype=np.int32)
    xblock_transform_graph = None
    xblock_transform_graph_out = None
    xblock_transform_graph_disabled = False

    for block_id, (shell_start, shell_stop, p0_block, p1_block) in enumerate(aux_blocks):
        shell_start = int(shell_start)
        shell_stop = int(shell_stop)
        p0_block = int(p0_block)
        p1_block = int(p1_block)
        if shell_stop <= shell_start:
            continue
        p_shells = np.arange(shell_start, shell_stop, dtype=np.int32)
        n_aux_shell_blk = int(p_shells.size)
        if n_aux_shell_blk == 0:
            continue

        naux_blk = int(p1_block - p0_block)
        if naux_blk <= 0:
            continue

        # Fast path: if the AO tensor block X(μ,ν,Pblk) fits in the user-provided scratch budget,
        # materialize it and use large GEMMs for the active transform. This is typically much
        # faster than the per-shell-pair digest, and still streams over aux blocks.
        x_block_bytes = int(nao) * int(nao) * int(naux_blk) * 8
        if strategy == "digest":
            use_x_block = False
        elif strategy == "x_block":
            if x_block_bytes > int(max_tile_bytes):
                raise RuntimeError(
                    "strategy='x_block' requested, but x-block scratch exceeds max_tile_bytes "
                    f"(required={int(x_block_bytes)}B, budget={int(max_tile_bytes)}B)"
                )
            use_x_block = True
        else:
            use_x_block = x_block_bytes <= int(max_tile_bytes)
        if use_x_block:
            if xblock_plan is None:
                xblock_plan = _get_cached_df_stream_xblock_plan(
                    ao_basis,
                    aux_basis,
                    aux_block_naux=aux_block_naux,
                    base_plan=plan,
                    stream=stream,
                    profile=profile,
                )

            if not (0 <= int(block_id) < int(len(xblock_plan.blocks))):
                raise RuntimeError("cached xblock plan missing expected aux block")
            blk = xblock_plan.blocks[int(block_id)]
            if int(blk.p0) != int(p0_block) or int(blk.p1) != int(p1_block):
                raise RuntimeError("cached xblock plan mismatch (aux block boundaries changed)")

            if profile is not None:
                prof = profile.setdefault("yt_stream", {})
                prof["use_x_block"] = True

            if X_blk_scratch is None:
                if int(xblock_scratch_naux) <= 0:
                    raise RuntimeError("internal error: xblock_scratch_naux must be > 0 when use_x_block is True")
                X_blk_scratch = cp.empty((nao, nao, int(xblock_scratch_naux)), dtype=cp.float64)
            X_blk = X_blk_scratch

            for bp in blk.batches:
                bytes_per_task = int(bp.nAB) * int(bp.nP) * 8
                if bytes_per_task <= 0:
                    continue
                max_tasks = max(1, max_tile_bytes // bytes_per_task)
                nt = int(bp.ntasks)
                if nt == 0:
                    continue

                start_k = cp.cuda.Event() if profile is not None else None
                end_k = cp.cuda.Event() if profile is not None else None
                start_sc = cp.cuda.Event() if profile is not None else None
                end_sc = cp.cuda.Event() if profile is not None else None
                s0 = cp.cuda.get_current_stream()

                for i0 in range(0, nt, max_tasks):
                    i1 = min(nt, i0 + max_tasks)
                    sub_tasks = TaskList(
                        task_spAB=bp.kernel_tasks.task_spAB[i0:i1],
                        task_spCD=bp.kernel_tasks.task_spCD[i0:i1],
                    )
                    sub_batch = KernelBatch(
                        task_idx=dummy_task_idx,
                        kernel_tasks=sub_tasks,
                        kernel_class_id=bp.kernel_class_id,
                        transpose=bool(bp.transpose),
                    )

                    if start_k is not None and end_k is not None:
                        start_k.record(s0)
                    tile = run_kernel_batch_spd(
                        sub_batch,
                        dbasis=dbasis,
                        dsp=dsp,
                        pt=pt,
                        stream=stream,
                        threads=threads,
                        mode=mode,
                        work_small_max=work_small_max,
                        work_large_min=work_large_min,
                        blocks_per_task=blocks_per_task,
                        profile=dispatch_profile,
                    )
                    if start_k is not None and end_k is not None:
                        end_k.record(s0)
                        end_k.synchronize()
                        k_ms = float(cp.cuda.get_elapsed_time(start_k, end_k))
                        prof = profile.setdefault("yt_stream", {})
                        prof["kernel_ms"] = float(prof.get("kernel_ms", 0.0)) + k_ms
                        key = (
                            f"{int(bp.kernel_class_id)}|tr={int(bp.transpose)}|native={int(bp.native)}|nAB={int(bp.nAB)}|nP={int(bp.nP)}"
                        )
                        row = prof.setdefault("batches", {}).setdefault(key, {"ms": 0.0, "ntasks": 0})
                        row["ms"] = float(row.get("ms", 0.0)) + k_ms
                        row["ntasks"] = int(row.get("ntasks", 0)) + int(i1 - i0)

                    if start_sc is not None and end_sc is not None:
                        start_sc.record(s0)
                    _ext.scatter_df_int3c2e_tiles_inplace_device(
                        tile.ravel(),
                        bp.a0_dev[i0:i1],
                        bp.b0_dev[i0:i1],
                        bp.p0_dev[i0:i1],
                        int(nao),
                        int(xblock_scratch_naux),
                        int(tile.shape[1]),
                        int(bp.nB),
                        int(tile.shape[2]),
                        X_blk.ravel(),
                        int(threads),
                        int(sptr),
                        False,
                    )
                    if start_sc is not None and end_sc is not None:
                        end_sc.record(s0)
                        end_sc.synchronize()
                        sc_ms = float(cp.cuda.get_elapsed_time(start_sc, end_sc))
                        prof = profile.setdefault("yt_stream", {})
                        prof["scatter_ms"] = float(prof.get("scatter_ms", 0.0)) + sc_ms

            # Transform X_blk to pair space: YT[P, pq] = (C^T X_P C)_{pq}.
            start_tr = cp.cuda.Event() if profile is not None else None
            end_tr = cp.cuda.Event() if profile is not None else None
            s0 = cp.cuda.get_current_stream()
            if start_tr is not None and end_tr is not None:
                start_tr.record(s0)
            L_full_blk = None
            if bool(graph_capture) and not bool(xblock_transform_graph_disabled):
                if xblock_transform_graph is None:
                    def _capture_body():
                        nonlocal xblock_transform_graph_out
                        xblock_transform_graph_out = active_Lfull_from_B(X_blk, C_active)  # (nops, xblock_scratch_naux)

                    ok, gobj, reason = _capture_cuda_graph(s0, _capture_body)
                    if ok:
                        xblock_transform_graph = gobj
                        if profile is not None:
                            prof = profile.setdefault("yt_stream", {})
                            prof["graph_capture_used"] = True
                    else:
                        xblock_transform_graph_disabled = True
                        if profile is not None:
                            prof = profile.setdefault("yt_stream", {})
                            prof["graph_capture_used"] = False
                            prof.setdefault("graph_capture_error", str(reason))

                if xblock_transform_graph is not None:
                    try:
                        _launch_cuda_graph(xblock_transform_graph, s0)
                    except Exception as e:  # pragma: no cover - runtime dependent
                        xblock_transform_graph_disabled = True
                        xblock_transform_graph = None
                        if profile is not None:
                            prof = profile.setdefault("yt_stream", {})
                            prof["graph_capture_used"] = False
                            prof.setdefault("graph_capture_error", str(e))
                    else:
                        if xblock_transform_graph_out is None:
                            raise RuntimeError("internal error: missing graph transform output buffer")
                        L_full_blk = xblock_transform_graph_out

            if L_full_blk is None:
                L_full_blk = active_Lfull_from_B(X_blk, C_active)  # (nops, xblock_scratch_naux)

            if start_tr is not None and end_tr is not None:
                end_tr.record(s0)
                end_tr.synchronize()
                prof = profile.setdefault("yt_stream", {})
                prof["transform_ms"] = float(prof.get("transform_ms", 0.0)) + float(cp.cuda.get_elapsed_time(start_tr, end_tr))
            YT[p0_block:p1_block, :] = L_full_blk[:, : int(naux_blk)].T
            continue

        if profile is not None:
            prof = profile.setdefault("yt_stream", {})
            prof["use_digest"] = True

        # Tasks: all (AO shell pair, aux shell) combinations for this aux-shell block.
        task_ab = np.repeat(ab_idx, n_aux_shell_blk)
        task_cd = (nsp_ao + np.tile(p_shells, int(ab_idx.size))).astype(np.int32, copy=False)
        tasks = TaskList(task_spAB=task_ab, task_spCD=task_cd)

        batches = plan_kernel_batches_spd(tasks, shell_pairs=sp_all, shell_l=shell_l)
        for batch in batches:
            # Chunk batches to bound tile memory.
            la, lb, lc, ld = decode_eri_class_id(int(batch.kernel_class_id))
            if batch.transpose:
                la, lb, lc, ld = lc, ld, la, lb
            nAB = int(ncart(int(la))) * int(ncart(int(lb)))
            nCD = int(ncart(int(lc))) * int(ncart(int(ld)))
            bytes_per_task = int(nAB * nCD) * 8
            if bytes_per_task <= 0:
                raise RuntimeError("invalid tile shape in streamed DF (bytes_per_task <= 0)")

            nt = int(batch.task_idx.shape[0])
            if nt == 0:
                continue

            # The original task list is AB-major, and within a fixed (la,lb,lc,ld) class batch the
            # per-AB task count should be constant (the number of aux shells in this aux block with
            # the batch's lc). Use this to reshape tiles and run one batched GEMM per aux shell,
            # instead of a GEMM per AO shell pair.
            batch_task_idx = np.asarray(batch.task_idx, dtype=np.int32).ravel()
            spab_full = np.asarray(tasks.task_spAB[batch_task_idx], dtype=np.int32).ravel()
            if spab_full.size != nt:
                raise RuntimeError("unexpected spab_full size in streamed DF")
            # Detect tasks-per-AB (m): run length of the first AB in the batch.
            m = 1
            while m < nt and int(spab_full[m]) == int(spab_full[0]):
                m += 1
            if m <= 0 or (nt % m) != 0:
                raise RuntimeError("unexpected task grouping in streamed DF (ntasks not divisible by tasks-per-AB)")
            n_ab_total = nt // m

            # Bound peak memory: tile + K_stack + one-P out buffer.
            bytes_per_ab_peak = int(m) * int(bytes_per_task) + int(nops) * int(nAB) * 8 + int(nops) * int(nCD) * 8
            if bytes_per_ab_peak <= 0:
                raise RuntimeError("invalid bytes_per_ab_peak in streamed DF")
            max_ab = max(1, max_tile_bytes // bytes_per_ab_peak)

            for ab0 in range(0, n_ab_total, max_ab):
                ab1 = min(n_ab_total, ab0 + max_ab)
                t0 = int(ab0 * m)
                t1 = int(ab1 * m)
                sub_idx = batch.task_idx[t0:t1]
                sub_tasks = TaskList(task_spAB=batch.kernel_tasks.task_spAB[t0:t1], task_spCD=batch.kernel_tasks.task_spCD[t0:t1])
                sub_batch = KernelBatch(
                    task_idx=np.asarray(sub_idx, dtype=np.int32),
                    kernel_tasks=sub_tasks,
                    kernel_class_id=batch.kernel_class_id,
                    transpose=batch.transpose,
                )

                tile = run_kernel_batch_spd(
                    sub_batch,
                    dbasis=dbasis,
                    dsp=dsp,
                    pt=pt,
                    stream=stream,
                    threads=threads,
                    mode=mode,
                    work_small_max=work_small_max,
                    work_large_min=work_large_min,
                    blocks_per_task=blocks_per_task,
                    profile=dispatch_profile,
                )

                n_ab_chunk = int(ab1 - ab0)
                if int(tile.shape[0]) != int(n_ab_chunk * m):
                    raise RuntimeError("unexpected tile task count in streamed DF")
                if int(tile.shape[1]) != int(nAB) or int(tile.shape[2]) != int(nCD):
                    raise RuntimeError("unexpected tile shape in streamed DF")
                tile4 = tile.reshape((n_ab_chunk, int(m), int(nAB), int(nCD)))

                spab_ab = np.asarray(spab_full[t0:t1:m], dtype=np.int32).ravel()
                if int(spab_ab.size) != n_ab_chunk:
                    raise RuntimeError("unexpected spab_ab size in streamed DF")

                K_list: list[cp.ndarray] = []
                for spab_id in spab_ab.tolist():
                    spab_id = int(spab_id)
                    K_AB_T = k_ab_t_cache.get(spab_id)
                    if K_AB_T is None:
                        A = int(sp_all.sp_A[spab_id])
                        B = int(sp_all.sp_B[spab_id])
                        a0 = int(ao_basis.shell_ao_start[A])
                        b0 = int(ao_basis.shell_ao_start[B])
                        nA = int(ncart(int(ao_l[A])))
                        nB = int(ncart(int(ao_l[B])))
                        CA = C_active[a0 : a0 + nA, :]
                        CB = C_active[b0 : b0 + nB, :]
                        K_AB = build_pair_coeff_ordered(CA, CB, same_shell=(A == B))  # (nAB, nops)
                        K_AB_T = cp.ascontiguousarray(K_AB.T)  # (nops, nAB)
                        k_ab_t_cache[spab_id] = K_AB_T
                    K_list.append(K_AB_T)
                K_stack = cp.stack(K_list, axis=0)  # (n_ab_chunk, nops, nAB)

                # P-shell order is identical for each AB; take it from the first AB in this chunk.
                spcd_first = np.asarray(tasks.task_spCD[np.asarray(sub_idx, dtype=np.int32).ravel()[:m]], dtype=np.int32).ravel()
                P_shell = (spcd_first - np.int32(nsp_ao)).astype(np.int32, copy=False)
                p0_list = np.asarray(aux_basis.shell_ao_start, dtype=np.int32).ravel()[P_shell]
                p0_list_dev = cp.ascontiguousarray(cp.asarray(p0_list, dtype=cp.int32))

                # Batch over multiple P-shells at once to reduce GEMM launch overhead.
                # Keep the temporary output within `max_tile_bytes` along with already
                # materialized `tile4` and `K_stack`.
                bytes_out_per_j = int(n_ab_chunk) * int(nops) * int(nCD) * 8
                bytes_tile = int(n_ab_chunk) * int(m) * int(bytes_per_task)
                bytes_k = int(n_ab_chunk) * int(nops) * int(nAB) * 8
                avail = int(max_tile_bytes) - int(bytes_tile + bytes_k)
                if avail <= 0 or bytes_out_per_j <= 0:
                    j_chunk = 1
                else:
                    j_chunk = max(1, int(avail // bytes_out_per_j))
                j_chunk = min(int(m), int(j_chunk))

                for j0 in range(0, int(m), int(j_chunk)):
                    j1 = min(int(m), j0 + int(j_chunk))
                    tile_j = tile4[:, j0:j1, :, :]  # (n_ab_chunk, j_chunk, nAB, nCD)
                    out = cp.matmul(K_stack[:, None, :, :], tile_j)  # (n_ab_chunk, j_chunk, nops, nCD)
                    out_sum = out.sum(axis=0)  # (j_chunk, nops, nCD)
                    out_sum = cp.ascontiguousarray(out_sum)
                    p0_dev = p0_list_dev[int(j0) : int(j1)]
                    if hasattr(_ext, "scatter_add_df_yt_tiles_inplace_device"):
                        _ext.scatter_add_df_yt_tiles_inplace_device(
                            out_sum.ravel(),
                            p0_dev,
                            int(naux),
                            int(nops),
                            int(nCD),
                            YT.ravel(),
                            int(threads),
                            int(sptr),
                            False,
                        )
                    else:
                        for jj in range(int(j1 - j0)):
                            p0 = int(p0_list[int(j0 + jj)])
                            YT[p0 : p0 + int(nCD), :] += out_sum[int(jj)].T

    if profile is not None:
        prof = profile.setdefault("yt_stream", {})
        prof.setdefault("strategy_requested", str(strategy))
        prof.setdefault("use_x_block", False)
        prof.setdefault("use_digest", False)
        prof.setdefault("kernel_ms", 0.0)
        prof.setdefault("scatter_ms", 0.0)
        prof.setdefault("transform_ms", 0.0)
        prof.setdefault("batches", {})
        prof.setdefault("kernel_dispatch", {})
        if "strategy" not in prof:
            use_x = bool(prof.get("use_x_block", False))
            use_d = bool(prof.get("use_digest", False))
            if use_x and use_d:
                prof["strategy"] = "mixed"
            elif use_x:
                prof["strategy"] = "x_block"
            elif use_d:
                prof["strategy"] = "digest"

    return YT


def active_Lfull_streamed_basis(
    ao_basis,
    aux_basis,
    C_active,
    *,
    stream=None,
    backend: str = "gpu_rys",
    threads: int = 256,
    mode: str = "auto",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 4,
    aux_block_naux: int = 256,
    max_tile_bytes: int = 256 * 1024 * 1024,
    strategy: str = "auto",
    ao_contract_mode: str = "auto",
    graph_capture: bool = False,
    profile: dict | None = None,
    out=None,
):
    """Compute active-space DF factors L_full[pq, Q] using a streaming approach on the GPU.

    Calculates the density fitting factors `L_{pq,Q} = (pq|Q) L_{QQ}^{-1/2}` without fully
    materializing the 3-center integral tensor `X(μν, P)` or the transformed `B(μν, Q)`.
    Instead, it processes the auxiliary basis in blocks to keep memory usage low.

    Parameters
    ----------
    ao_basis : object
        The atomic orbital basis set definition.
    aux_basis : object
        The auxiliary basis set definition.
    C_active : array-like
        Active space MO coefficients. Shape `(nao, norb)`.
    stream : cupy.cuda.Stream | None, optional
        CUDA stream for asynchronous execution.
    backend : str, default='gpu_rys'
        Integrals backend to use ('gpu_ss', 'gpu_sp', 'gpu_rys').
    threads : int, default=256
        Block size for CUDA kernels.
    mode : str, default='auto'
        Execution strategy ('block', 'warp', etc.).
    work_small_max : int, default=512
        Threshold for small work dispatch.
    work_large_min : int, default=200000
        Threshold for large work dispatch.
    blocks_per_task : int, default=4
        Blocks per task for multiblock kernels.
    aux_block_naux : int, default=256
        Block size for auxiliary functions in the streaming loop.
    max_tile_bytes : int, default=268435456
        Maximum size (in bytes) for temporary tile buffers.
    strategy : str, default='auto'
        Streaming strategy: 'auto', 'x_block', or 'digest'.
    ao_contract_mode : str, default='auto'
        AO contraction handling for full-X calls: 'auto', 'expanded', or 'native_contracted'.
    graph_capture : bool, default=False
        Best-effort CUDA Graph capture for repeated x-block transform launches.
        Falls back to normal launches if capture is unsupported for the active runtime/path.
    profile : dict | None, optional
        Dictionary for collecting performance profile data.
    out : cupy.ndarray | None, optional
        Output array of shape `(norb*norb, naux)`.

    Returns
    -------
    cupy.ndarray
        The computed DF factors `L_full` with shape `(norb*norb, naux)`.
    """

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for streamed DF active build") from e

    backend = str(backend).lower().strip()
    if backend not in ("gpu_ss", "gpu_sp", "gpu_rys"):
        raise ValueError("backend must be one of: 'gpu_ss', 'gpu_sp', 'gpu_rys'")

    with stream_ctx(stream):
        L = _get_cached_metric_cholesky(aux_basis, backend=backend, stream=None, profile=profile)

        YT = _active_YT_streamed_rys_basis(
            ao_basis,
            aux_basis,
            C_active,
            stream=None,
            threads=threads,
            mode=mode,
            work_small_max=work_small_max,
            work_large_min=work_large_min,
            blocks_per_task=blocks_per_task,
            aux_block_naux=aux_block_naux,
            max_tile_bytes=max_tile_bytes,
            strategy=str(strategy),
            ao_contract_mode=str(ao_contract_mode),
            graph_capture=bool(graph_capture),
            profile=profile,
        )

        import cupyx.scipy.linalg as cpx_linalg

        # L_full^T = L^{-1} Y^T
        start = None
        end = None
        if profile is not None:
            import cupy as cp

            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record(cp.cuda.get_current_stream())
        L_full_T = cpx_linalg.solve_triangular(L, YT, lower=True, trans="N", unit_diagonal=False, overwrite_b=True)
        if profile is not None and start is not None and end is not None:
            import cupy as cp

            end.record(cp.cuda.get_current_stream())
            end.synchronize()
            prof = profile.setdefault("solve_triangular", {})
            prof["ms"] = float(cp.cuda.get_elapsed_time(start, end))
        L_full = L_full_T.T
        if out is None:
            return cp.ascontiguousarray(L_full)

        if not isinstance(out, cp.ndarray):
            raise TypeError("out must be a CuPy ndarray")
        out_arr = out
        if out_arr.ndim != 2:
            raise ValueError("out must be a 2D CuPy array with shape (norb*norb, naux)")
        if out_arr.dtype != cp.float64:
            raise ValueError(f"out must have dtype float64 (got {out_arr.dtype})")
        shape = getattr(C_active, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError("C_active must have shape (nao, norb)")
        norb = int(shape[1])
        nops = int(norb) * int(norb)
        naux = int(L.shape[0])
        if tuple(out_arr.shape) != (nops, naux):
            raise ValueError(f"out has shape {tuple(out_arr.shape)}, expected {(nops, naux)}")
        if not bool(out_arr.flags.c_contiguous):
            raise ValueError("out must be C-contiguous")
        cp.copyto(out_arr, L_full)
        return out_arr


def plan_aux_blocks_cart(aux_basis, max_block_naux: int) -> list[tuple[int, int, int, int]]:
    """Partition the auxiliary basis into contiguous blocks aligned with shell boundaries.

    Helper function to divide the auxiliary basis into manageable chunks for streaming
    operations. Ensures no shell is split across blocks.

    Parameters
    ----------
    aux_basis : BasisCartSoA
        A packed Cartesian auxiliary basis set. Must provide `shell_ao_start` and `shell_l`.
    max_block_naux : int
        Maximum number of auxiliary basis functions per block.

    Returns
    -------
    list[tuple[int, int, int, int]]
        A list of block descriptors. Each tuple contains:
        `(shell_start, shell_stop, p0, p1)` where:
        - `shell_start`, `shell_stop`: Range of shell indices [start, stop).
        - `p0`, `p1`: Range of AO function indices [start, stop).
    """

    if not hasattr(aux_basis, "shell_ao_start") or not hasattr(aux_basis, "shell_l"):
        raise TypeError("aux_basis must provide shell_ao_start and shell_l")
    max_block_naux = int(max_block_naux)
    if max_block_naux <= 0:
        raise ValueError("max_block_naux must be > 0")

    shell_ao_start = np.asarray(aux_basis.shell_ao_start, dtype=np.int64).ravel()
    shell_l = np.asarray(aux_basis.shell_l, dtype=np.int64).ravel()
    if shell_ao_start.shape != shell_l.shape:
        raise ValueError("aux_basis.shell_ao_start and shell_l must have identical shape")

    nshell = int(shell_l.size)
    if nshell == 0:
        return []

    nfunc = np.asarray([ncart(int(l)) for l in shell_l], dtype=np.int64)
    shell_ao_end = shell_ao_start + nfunc
    if int(shell_ao_start[0]) != 0:
        raise ValueError("expected aux shell_ao_start[0] == 0")
    if np.any(shell_ao_start[1:] != shell_ao_end[:-1]):
        raise ValueError("aux shells must be contiguous in AO ordering (unexpected gaps/overlaps)")

    blocks: list[tuple[int, int, int, int]] = []
    shell_start = 0
    p0 = int(shell_ao_start[0])
    acc = 0
    for s in range(nshell):
        nf = int(nfunc[s])
        if acc > 0 and acc + nf > max_block_naux:
            shell_stop = s
            p1 = int(shell_ao_start[s])
            blocks.append((int(shell_start), int(shell_stop), int(p0), int(p1)))
            shell_start = s
            p0 = int(shell_ao_start[s])
            acc = 0
        acc += nf

    blocks.append((int(shell_start), int(nshell), int(p0), int(shell_ao_end[-1])))
    return blocks


def int3c2e_block(ao_basis, aux_basis, p0: int, p1: int, *, stream=None, backend: str = "gpu_rys"):
    """Compute a specific block of 3-center 2-electron (3c2e) integrals on the GPU.

    Calculates the slice `(μν|P)` where `P` ranges from `p0` to `p1`. The resulting
    tensor has shape `(nao, nao, p1-p0)`.

    Parameters
    ----------
    ao_basis : object
        The atomic orbital basis set definition.
    aux_basis : object
        The auxiliary basis set definition.
    p0 : int
        Starting index of the auxiliary function block (inclusive).
    p1 : int
        Ending index of the auxiliary function block (exclusive).
    stream : cupy.cuda.Stream | None, optional
        CUDA stream for asynchronous execution.
    backend : str, default='gpu_rys'
        Integrals backend to use.

    Returns
    -------
    cupy.ndarray
        The computed integral block `X[:, :, p0:p1]`.

    Notes
    -----
    - For GPU backends, `p0` and `p1` must align with auxiliary shell boundaries.
    - Use `plan_aux_blocks_cart` to generate valid block indices.
    - Designed for streaming operations to avoid large memory allocations.
    """

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for DF int3c2e_block") from e

    p0 = int(p0)
    p1 = int(p1)
    if p0 < 0 or p1 < 0 or p1 < p0:
        raise ValueError("expected 0 <= p0 <= p1")

    backend = str(backend).lower().strip()
    if backend not in ("gpu_ss", "gpu_sp", "gpu_rys"):
        raise ValueError("backend must be one of: 'gpu_ss', 'gpu_sp', 'gpu_rys'")

    with stream_ctx(stream):
        if backend in ("gpu_sp", "gpu_rys"):
            from .gpu import df_int3c2e_rys_device_block, has_cuda_ext

            if not has_cuda_ext():
                raise RuntimeError("cuERI CUDA extension not available; build via `python -m asuka.cueri.build_cuda_ext`")

            shell_ao_start = np.asarray(aux_basis.shell_ao_start, dtype=np.int64).ravel()
            shell_l = np.asarray(aux_basis.shell_l, dtype=np.int64).ravel()
            nfunc = np.asarray([ncart(int(l)) for l in shell_l], dtype=np.int64)
            shell_ao_end = shell_ao_start + nfunc
            naux = int(shell_ao_end[-1]) if shell_ao_end.size else 0
            if p1 > naux:
                raise ValueError(f"p1={p1} out of range for naux={naux}")
            if p0 == p1:
                ao_start = np.asarray(ao_basis.shell_ao_start, dtype=np.int64).ravel()
                from .basis_utils import shell_nfunc_cart

                ao_nfunc = np.asarray(shell_nfunc_cart(ao_basis), dtype=np.int64)
                nao = int(np.max(ao_start + ao_nfunc)) if ao_start.size else 0
                return cp.empty((nao, nao, 0), dtype=cp.float64)

            start_matches = np.nonzero(shell_ao_start == p0)[0]
            if start_matches.size != 1:
                raise ValueError("p0 must align to an aux shell boundary")
            shell_start = int(start_matches[0])

            shell_stop = int(np.searchsorted(shell_ao_start, p1, side="left"))
            if shell_stop < int(shell_ao_start.size):
                if int(shell_ao_start[shell_stop]) != p1:
                    raise ValueError("p1 must align to an aux shell boundary")
            else:
                if int(shell_ao_end[-1]) != p1:
                    raise ValueError("p1 must align to an aux shell boundary")

            return df_int3c2e_rys_device_block(
                ao_basis,
                aux_basis,
                aux_shell_start=shell_start,
                aux_shell_stop=shell_stop,
                stream=None,
            )

        # Fallback for s-only GPU baseline: materialize then slice.
        X = int3c2e_basis(ao_basis, aux_basis, stream=None, backend=backend)
        X = cp.asarray(X, dtype=cp.float64)
        if X.ndim != 3:
            raise RuntimeError(f"unexpected X ndim={X.ndim}")
        naux = int(X.shape[2])
        if p1 > naux:
            raise ValueError(f"p1={p1} out of range for naux={naux}")
        return X[:, :, p0:p1]


__all__ = [
    "active_Lfull_from_B",
    "active_Lfull_from_cached_B_whitened",
    "active_Lfull_streamed_basis",
    "cholesky_metric",
    "int3c2e_basis",
    "int3c2e_block",
    "metric_2c2e_basis",
    "plan_aux_blocks_cart",
    "recommended_auxbasis",
    "whiten_3c2e",
]
