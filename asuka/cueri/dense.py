from __future__ import annotations

import numpy as np
import weakref
from .cart import ncart
from .basis_utils import shell_nfunc_cart
from .eri_utils import build_pair_coeff_ordered, build_pair_coeff_ordered_mixed, build_pair_coeff_packed
from .shell_pairs import build_shell_pairs_l_order
from .tasks import (
    TaskList,
    build_tasks_screened,
    build_tasks_screened_sorted_q,
    decode_eri_class_id,
    eri_class_id,
    group_tasks_by_class,
    group_tasks_by_spab,
    with_task_class_id,
)
from .tile_eval import iter_tile_batches_spd
from .stream import stream_ctx, with_stream


_DENSE_GPU_PUWX_BUILDER_CACHE_MAX = 4
_dense_gpu_puwx_builder_cache: dict[tuple, tuple[weakref.ref, object]] = {}
_DENSE_GPU_ACTIVE_BUILDER_CACHE_MAX = 4
_dense_gpu_active_builder_cache: dict[tuple, tuple[weakref.ref, object]] = {}


def _resolve_active_dense_gpu_builder(
    ao_basis,
    *,
    threads: int,
    max_tile_bytes: int,
    eps_ao: float,
    sp_Q,
    algorithm: str = "auto",
    mode: str = "auto",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
    boys: str = "ref",
):
    """Resolve (and cache) a CuERI active-space dense GPU builder for a basis/config tuple."""

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for GPU dense builder") from e

    try:
        from .active_space_dense_gpu import CuERIActiveSpaceDenseGPUBuilder  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("cuERI active-space GPU builder is required for dense build") from e

    eps_ao_f = float(eps_ao)
    threads_i = int(threads)
    max_tile_bytes_i = int(max_tile_bytes)
    spq_id = 0 if sp_Q is None else int(id(sp_Q))
    key = (
        int(cp.cuda.runtime.getDevice()),
        int(id(ao_basis)),
        int(threads_i),
        int(max_tile_bytes_i),
        float(eps_ao_f),
        int(spq_id),
        str(algorithm),
        str(mode),
        int(work_small_max),
        int(work_large_min),
        int(blocks_per_task),
        str(boys),
    )

    hit = _dense_gpu_active_builder_cache.get(key)
    if hit is not None:
        basis_ref, bld = hit
        if basis_ref() is ao_basis:
            return bld
        del _dense_gpu_active_builder_cache[key]

    builder = CuERIActiveSpaceDenseGPUBuilder(
        ao_basis=ao_basis,
        threads=int(threads_i),
        max_tile_bytes=int(max_tile_bytes_i),
        eps_ao=float(eps_ao_f),
        sp_Q=None if sp_Q is None else np.asarray(sp_Q, dtype=np.float64),
        stream=None,
        algorithm=str(algorithm),
        mode=str(mode),
        work_small_max=int(work_small_max),
        work_large_min=int(work_large_min),
        blocks_per_task=int(blocks_per_task),
        boys=str(boys),
        ao_rep="cart",
    )
    _dense_gpu_active_builder_cache[key] = (weakref.ref(ao_basis), builder)
    while len(_dense_gpu_active_builder_cache) > _DENSE_GPU_ACTIVE_BUILDER_CACHE_MAX:
        _dense_gpu_active_builder_cache.pop(next(iter(_dense_gpu_active_builder_cache)))
    return builder


def _filter_builder_tasks_by_eps_mo(builder, C_active, *, eps_mo: float):
    """Return a builder-like view with tasks additionally screened by eps_mo.

    The additional screening criterion is
      Q_ab * Q_cd * M_ab * M_cd >= eps_mo
    where M_ab is the shell-pair MO magnitude proxy used for packed eps_mo screening.
    """

    eps_mo_f = float(eps_mo)
    if eps_mo_f <= 0.0:
        return builder

    tasks = getattr(builder, "tasks", None)
    sp = getattr(builder, "sp", None)
    basis = getattr(builder, "ao_basis", None)
    Q_np = np.asarray(getattr(builder, "Q_np", np.zeros((0,), dtype=np.float64)), dtype=np.float64).ravel()
    if tasks is None or sp is None or basis is None or int(getattr(tasks, "ntasks", 0)) == 0:
        return builder
    if Q_np.size == 0:
        return builder

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for eps_mo task filtering on GPU dense path") from e

    C_active_dev = cp.asarray(C_active, dtype=cp.float64)
    C_active_dev = cp.ascontiguousarray(C_active_dev)
    C_abs_host = cp.asnumpy(cp.abs(C_active_dev))

    shell_l = np.asarray(basis.shell_l, dtype=np.int32).ravel()
    shell_ao_start = np.asarray(basis.shell_ao_start, dtype=np.int32).ravel()
    n_shell = int(shell_l.size)
    M_shell = np.zeros((n_shell,), dtype=np.float64)
    for S in range(n_shell):
        a0 = int(shell_ao_start[S])
        n = int(ncart(int(shell_l[S])))
        if n <= 0:
            continue
        M_shell[S] = float(np.max(C_abs_host[a0 : a0 + n, :]))

    spA = np.asarray(sp.sp_A, dtype=np.int32).ravel()
    spB = np.asarray(sp.sp_B, dtype=np.int32).ravel()
    M_sp = M_shell[spA] * M_shell[spB]

    task_ab = np.asarray(tasks.task_spAB, dtype=np.int32).ravel()
    task_cd = np.asarray(tasks.task_spCD, dtype=np.int32).ravel()
    if task_ab.size == 0:
        return builder

    qprod = Q_np[task_ab] * Q_np[task_cd]
    keep = qprod * M_sp[task_ab] * M_sp[task_cd] >= eps_mo_f
    if bool(np.all(keep)):
        return builder

    task_ab_f = np.asarray(task_ab[keep], dtype=np.int32)
    task_cd_f = np.asarray(task_cd[keep], dtype=np.int32)
    tasks_f = TaskList(task_spAB=task_ab_f, task_spCD=task_cd_f)
    if tasks_f.ntasks > 0:
        tasks_f = with_task_class_id(tasks_f, sp, shell_l)
        assert tasks_f.task_class_id is not None
        perm, class_ids, offsets = group_tasks_by_class(tasks_f.task_class_id)
        task_ab_sorted = np.asarray(tasks_f.task_spAB[perm], dtype=np.int32)
        task_cd_sorted = np.asarray(tasks_f.task_spCD[perm], dtype=np.int32)
        nsp = int(spA.shape[0])
        perm_ab, ab_offsets = group_tasks_by_spab(tasks_f.task_spAB, nsp)
        task_cd_by_ab = np.asarray(tasks_f.task_spCD[perm_ab], dtype=np.int32)
    else:
        perm = np.zeros((0,), dtype=np.int32)
        class_ids = np.zeros((0,), dtype=np.int32)
        offsets = np.asarray([0], dtype=np.int32)
        task_ab_sorted = np.zeros((0,), dtype=np.int32)
        task_cd_sorted = np.zeros((0,), dtype=np.int32)
        perm_ab = np.zeros((0,), dtype=np.int32)
        ab_offsets = np.zeros((int(spA.shape[0]) + 1,), dtype=np.int32)
        task_cd_by_ab = np.zeros((0,), dtype=np.int32)

    from types import SimpleNamespace

    b_exec = SimpleNamespace(**vars(builder))
    b_exec.tasks = tasks_f
    b_exec.perm = perm
    b_exec.class_ids = class_ids
    b_exec.offsets = offsets
    b_exec.task_ab = task_ab_sorted
    b_exec.task_cd = task_cd_sorted
    b_exec.perm_ab = perm_ab
    b_exec.ab_offsets = ab_offsets
    b_exec.task_cd_by_ab = task_cd_by_ab
    return b_exec


def _build_pair_coeff_ordered_mixed_batch(CA_p, CB_p, CA_u, CB_u, *, same_shell: np.ndarray | None):
    """Batch analogue of :func:`asuka.cueri.eri_utils.build_pair_coeff_ordered_mixed` (ordered mixed pairs).

    Builds coefficients for the mixed ordered pair space:
      (p,u) with p in [0,nmo), u in [0,ncas)
    using the convention `pu = p*ncas + u`.

    Parameters
    ----------
    CA_p, CB_p
        CuPy arrays with shapes (nt, nA, nmo) and (nt, nB, nmo).
    CA_u, CB_u
        CuPy arrays with shapes (nt, nA, ncas) and (nt, nB, ncas).
    same_shell
        Boolean vector with shape (nt,) where True indicates A==B (no swapped AO-product term).

    Returns
    -------
    K
        CuPy array with shape (nt, nA*nB, nmo*ncas).
    """

    import cupy as cp

    CA_p = cp.asarray(CA_p)
    CB_p = cp.asarray(CB_p)
    CA_u = cp.asarray(CA_u)
    CB_u = cp.asarray(CB_u)
    if CA_p.ndim != 3 or CB_p.ndim != 3 or CA_u.ndim != 3 or CB_u.ndim != 3:
        raise ValueError("CA_p/CB_p/CA_u/CB_u must be 3D arrays with shape (nt, nAO_in_shell, n_orb)")
    if int(CA_p.shape[0]) != int(CB_p.shape[0]) or int(CA_p.shape[0]) != int(CA_u.shape[0]) or int(CA_p.shape[0]) != int(CB_u.shape[0]):
        raise ValueError("mixed pair coeffs require matching batch size (nt)")
    if int(CA_p.shape[1]) != int(CA_u.shape[1]) or int(CB_p.shape[1]) != int(CB_u.shape[1]):
        raise ValueError("mixed pair coeffs require matching AO dims (CA_p vs CA_u, CB_p vs CB_u)")
    if int(CA_p.shape[2]) != int(CB_p.shape[2]):
        raise ValueError("CA_p and CB_p must have the same nmo")
    if int(CA_u.shape[2]) != int(CB_u.shape[2]):
        raise ValueError("CA_u and CB_u must have the same ncas")

    nt, nA, nmo = map(int, CA_p.shape)
    nB = int(CB_p.shape[1])
    ncas = int(CA_u.shape[2])

    # K[t,a,b,p,u] = CA_p[t,a,p] * CB_u[t,b,u]
    K = cp.einsum("tap,tbu->tabpu", CA_p, CB_u, optimize=True)
    if same_shell is None:
        K = K + cp.einsum("tbp,tau->tabpu", CB_p, CA_u, optimize=True)
    else:
        same = np.asarray(same_shell, dtype=bool).ravel()
        if same.shape != (nt,):
            raise ValueError(f"same_shell must have shape ({nt},), got {same.shape}")
        if not bool(np.all(same)):
            off = np.nonzero(~same)[0]
            if int(off.size):
                K[off] = K[off] + cp.einsum("tbp,tau->tabpu", CB_p[off], CA_u[off], optimize=True)

    return K.reshape((nt, nA * nB, nmo * ncas))


def _build_pair_coeff_ordered_batch(CA, CB, *, same_shell: np.ndarray | None):
    """Batch analogue of :func:`asuka.cueri.eri_utils.build_pair_coeff_ordered` (ordered pairs).

    Parameters
    ----------
    CA, CB
        CuPy arrays with shapes (nt, nA, norb) and (nt, nB, norb).
    same_shell
        Boolean vector with shape (nt,) where True indicates A==B (no swapped AO-product term).

    Returns
    -------
    K
        CuPy array with shape (nt, nA*nB, norb*norb).
    """

    import cupy as cp

    CA = cp.asarray(CA)
    CB = cp.asarray(CB)
    if CA.ndim != 3 or CB.ndim != 3:
        raise ValueError("CA/CB must be 3D arrays with shape (nt, nAO_in_shell, norb)")
    if int(CA.shape[0]) != int(CB.shape[0]) or int(CA.shape[2]) != int(CB.shape[2]):
        raise ValueError("CA/CB must agree on batch size (nt) and norb")

    nt, nA, norb = map(int, CA.shape)
    nB = int(CB.shape[1])

    # K[t,a,b,p,q] = CA[t,a,p] * CB[t,b,q]
    K = cp.einsum("tap,tbq->tabpq", CA, CB, optimize=True)
    if same_shell is None:
        K = K + cp.einsum("tbp,taq->tabpq", CB, CA, optimize=True)
    else:
        same = np.asarray(same_shell, dtype=bool).ravel()
        if same.shape != (nt,):
            raise ValueError(f"same_shell must have shape ({nt},), got {same.shape}")
        if not bool(np.all(same)):
            off = np.nonzero(~same)[0]
            if int(off.size):
                K[off] = K[off] + cp.einsum("tbp,taq->tabpq", CB[off], CA[off], optimize=True)

    return K.reshape((nt, nA * nB, norb * norb))


def _accum_KtB_ordered_blocked(out_eri_mat, K_AB, B, *, add_transpose: bool, pair_block: int) -> None:
    """Accumulate out_eri_mat += K_AB^T @ B, optionally adding transpose, in column blocks."""

    import cupy as cp

    n_pair = int(K_AB.shape[1])
    if n_pair == 0:
        return

    pair_block_i = int(pair_block)
    if pair_block_i <= 0 or pair_block_i >= n_pair:
        C = cp.matmul(K_AB.T, B)
        out_eri_mat += C
        if add_transpose:
            out_eri_mat += C.T
        return

    for j0 in range(0, n_pair, pair_block_i):
        j1 = min(n_pair, j0 + pair_block_i)
        Cblk = cp.matmul(K_AB.T, B[:, j0:j1])
        out_eri_mat[:, j0:j1] += Cblk
        if add_transpose:
            out_eri_mat[j0:j1, :] += Cblk.T


def _build_active_eri_mat_dense_rys_from_cached_direct(*, builder, C_active, out_eri_mat, profile: dict | None = None) -> None:
    """Direct cached dense executor (per-task AB/CD transforms), using dispatcher tile evaluation."""

    import time
    import cupy as cp

    if profile is not None:
        profile.clear()
        prof = profile.setdefault("cueri_dense_rys_cached", {})
        prof["plan_cache_hit"] = True
        t0 = time.perf_counter()
    else:
        prof = None
        t0 = None

    sp = getattr(builder, "sp", None)
    basis = getattr(builder, "ao_basis", None)
    class_ids = getattr(builder, "class_ids", None)
    offsets = getattr(builder, "offsets", None)
    task_ab_all = getattr(builder, "task_ab", None)
    task_cd_all = getattr(builder, "task_cd", None)
    dbasis = getattr(builder, "dbasis", None)
    dsp = getattr(builder, "dsp", None)
    pt = getattr(builder, "pair_tables", None)
    if any(x is None for x in (sp, basis, class_ids, offsets, task_ab_all, task_cd_all, dbasis, dsp, pt)):
        raise RuntimeError("builder is missing cached preprocessing/device artifacts")

    C_active = cp.asarray(C_active, dtype=cp.float64)
    C_active = cp.ascontiguousarray(C_active)
    if C_active.ndim != 2:
        raise ValueError("C_active must have shape (nao, norb)")
    nao, norb = map(int, C_active.shape)

    n_pair = int(norb) * int(norb)
    if out_eri_mat.shape != (n_pair, n_pair):
        raise ValueError(f"out_eri_mat must have shape ({n_pair},{n_pair}), got {out_eri_mat.shape}")
    out_eri_mat.fill(0)

    shell_ao_start = np.asarray(basis.shell_ao_start, dtype=np.int64)
    shell_l = np.asarray(basis.shell_l, dtype=np.int32)
    spA = np.asarray(sp.sp_A, dtype=np.int32)
    spB = np.asarray(sp.sp_B, dtype=np.int32)
    sp_a0 = shell_ao_start[spA]
    sp_b0 = shell_ao_start[spB]
    sp_same = spA == spB

    threads = int(getattr(builder, "threads", 256))
    max_tile_bytes_i = int(getattr(builder, "max_tile_bytes", 256 << 20))
    max_work_bytes = int(max_tile_bytes_i)
    mode = str(getattr(builder, "mode", "auto")).lower().strip()
    work_small_max = int(getattr(builder, "work_small_max", 512))
    work_large_min = int(getattr(builder, "work_large_min", 200_000))
    blocks_per_task = int(getattr(builder, "blocks_per_task", 8))
    boys = str(getattr(builder, "boys", "ref")).lower().strip()

    n_tasks_total = int(getattr(getattr(builder, "tasks", None), "ntasks", int(task_ab_all.shape[0])))
    arange_cache: dict[int, cp.ndarray] = {}

    def _arange(n: int):
        arr = arange_cache.get(int(n))
        if arr is None:
            arr = cp.arange(int(n), dtype=cp.int32)
            arange_cache[int(n)] = arr
        return arr

    for g in range(int(class_ids.shape[0])):
        j0 = int(offsets[g])
        j1 = int(offsets[g + 1])
        if j1 <= j0:
            continue

        task_group = TaskList(
            task_spAB=np.asarray(task_ab_all[j0:j1], dtype=np.int32),
            task_spCD=np.asarray(task_cd_all[j0:j1], dtype=np.int32),
        )
        if task_group.ntasks == 0:
            continue

        for tb in iter_tile_batches_spd(
            task_group,
            shell_pairs=sp,
            shell_l=shell_l,
            dbasis=dbasis,
            dsp=dsp,
            pt=pt,
            stream=None,
            threads=int(threads),
            mode=mode,
            work_small_max=int(work_small_max),
            work_large_min=int(work_large_min),
            blocks_per_task=int(blocks_per_task),
            max_tile_bytes=int(max_tile_bytes_i),
            boys=boys,
        ):
            tile = tb.tiles
            spab_batch = np.asarray(tb.task_spAB, dtype=np.int32)
            spcd_batch = np.asarray(tb.task_spCD, dtype=np.int32)
            nt = int(spab_batch.shape[0])
            if nt == 0:
                continue

            nAB = int(tile.shape[1])
            nCD = int(tile.shape[2])
            bytes_work_task = (int(nCD) * n_pair + 2 * int(nAB) * n_pair) * 8
            max_tasks = int(max(1, max_work_bytes // max(bytes_work_task, 1)))

            for i0 in range(0, nt, max_tasks):
                i1 = min(nt, i0 + max_tasks)
                spab = spab_batch[i0:i1]
                spcd = spcd_batch[i0:i1]
                tile_chunk = tile[i0:i1]
                m = int(spab.shape[0])
                if m == 0:
                    continue

                A0 = int(spA[int(spab[0])])
                B0 = int(spB[int(spab[0])])
                C0 = int(spA[int(spcd[0])])
                D0 = int(spB[int(spcd[0])])
                nA = int(ncart(int(shell_l[A0])))
                nB = int(ncart(int(shell_l[B0])))
                nC = int(ncart(int(shell_l[C0])))
                nD = int(ncart(int(shell_l[D0])))

                a0 = sp_a0[spab]
                b0 = sp_b0[spab]
                c0 = sp_a0[spcd]
                d0 = sp_b0[spcd]

                ia = cp.asarray(a0, dtype=cp.int32)[:, None] + _arange(nA)[None, :]
                ib = cp.asarray(b0, dtype=cp.int32)[:, None] + _arange(nB)[None, :]
                ic = cp.asarray(c0, dtype=cp.int32)[:, None] + _arange(nC)[None, :]
                id_ = cp.asarray(d0, dtype=cp.int32)[:, None] + _arange(nD)[None, :]

                CA = C_active[ia, :]
                CB = C_active[ib, :]
                CC = C_active[ic, :]
                CD = C_active[id_, :]

                same_ab = sp_same[spab]
                same_cd = sp_same[spcd]
                K_AB = _build_pair_coeff_ordered_batch(CA, CB, same_shell=same_ab)
                K_CD = _build_pair_coeff_ordered_batch(CC, CD, same_shell=same_cd)

                tmp = cp.matmul(tile_chunk, K_CD)
                out_eri_mat += K_AB.reshape((m * nAB, n_pair)).T @ tmp.reshape((m * nAB, n_pair))

                off = spab != spcd
                if bool(np.any(off)):
                    tile_off = tile_chunk[off]
                    K_AB_off = K_AB[off]
                    K_CD_off = K_CD[off]
                    noff = int(tile_off.shape[0])
                    if noff:
                        tmp2 = cp.matmul(tile_off.transpose((0, 2, 1)), K_AB_off)
                        out_eri_mat += K_CD_off.reshape((noff * nCD, n_pair)).T @ tmp2.reshape((noff * nCD, n_pair))

    if prof is not None and t0 is not None:
        cp.cuda.get_current_stream().synchronize()
        prof["t_eri_mat_s"] = float(time.perf_counter() - float(t0))
        prof["nao"] = int(nao)
        prof["norb"] = int(norb)
        prof["n_tasks"] = int(n_tasks_total)
        prof["threads"] = int(threads)
        prof["max_tile_bytes"] = int(max_tile_bytes_i)
        prof["algorithm"] = "direct"


def _build_active_eri_mat_dense_rys_from_cached_ab_group(*, builder, C_active, out_eri_mat, profile: dict | None = None) -> None:
    """AB-group dense executor using dispatcher tiles and one K_AB contraction per AB shell pair."""

    import time
    import cupy as cp

    if profile is not None:
        profile.clear()
        prof = profile.setdefault("cueri_dense_rys_cached", {})
        prof["plan_cache_hit"] = True
        t0 = time.perf_counter()
    else:
        prof = None
        t0 = None

    sp = getattr(builder, "sp", None)
    basis = getattr(builder, "ao_basis", None)
    dbasis = getattr(builder, "dbasis", None)
    dsp = getattr(builder, "dsp", None)
    pt = getattr(builder, "pair_tables", None)
    tasks = getattr(builder, "tasks", None)
    if any(x is None for x in (sp, basis, dbasis, dsp, pt, tasks)):
        raise RuntimeError("builder is missing cached preprocessing/device artifacts")

    ab_offsets = getattr(builder, "ab_offsets", None)
    task_cd_by_ab = getattr(builder, "task_cd_by_ab", None)
    if ab_offsets is None or task_cd_by_ab is None:
        perm_ab, ab_offsets = group_tasks_by_spab(tasks.task_spAB, nsp=int(np.asarray(sp.sp_A, dtype=np.int32).shape[0]))
        task_cd_by_ab = np.asarray(tasks.task_spCD[perm_ab], dtype=np.int32)

    C_active = cp.asarray(C_active, dtype=cp.float64)
    C_active = cp.ascontiguousarray(C_active)
    if C_active.ndim != 2:
        raise ValueError("C_active must have shape (nao, norb)")
    nao, norb = map(int, C_active.shape)

    n_pair = int(norb) * int(norb)
    if out_eri_mat.shape != (n_pair, n_pair):
        raise ValueError(f"out_eri_mat must have shape ({n_pair},{n_pair}), got {out_eri_mat.shape}")
    out_eri_mat.fill(0)

    shell_ao_start = np.asarray(basis.shell_ao_start, dtype=np.int64)
    shell_l = np.asarray(basis.shell_l, dtype=np.int32)
    spA = np.asarray(sp.sp_A, dtype=np.int32)
    spB = np.asarray(sp.sp_B, dtype=np.int32)
    sp_a0 = shell_ao_start[spA]
    sp_b0 = shell_ao_start[spB]
    nsp = int(spA.shape[0])

    threads = int(getattr(builder, "threads", 256))
    max_tile_bytes_i = int(getattr(builder, "max_tile_bytes", 256 << 20))
    max_work_bytes = int(max_tile_bytes_i)
    mode = str(getattr(builder, "mode", "auto")).lower().strip()
    work_small_max = int(getattr(builder, "work_small_max", 512))
    work_large_min = int(getattr(builder, "work_large_min", 200_000))
    blocks_per_task = int(getattr(builder, "blocks_per_task", 8))
    boys = str(getattr(builder, "boys", "ref")).lower().strip()
    pair_block = int(getattr(builder, "pair_block", 0))

    n_tasks_total = int(getattr(tasks, "ntasks", int(np.asarray(task_cd_by_ab).shape[0])))
    arange_cache: dict[int, cp.ndarray] = {}

    def _arange(n: int):
        arr = arange_cache.get(int(n))
        if arr is None:
            arr = cp.arange(int(n), dtype=cp.int32)
            arange_cache[int(n)] = arr
        return arr

    for spab in range(nsp):
        j0 = int(ab_offsets[spab])
        j1 = int(ab_offsets[spab + 1])
        if j1 <= j0:
            continue

        spcd_all = np.asarray(task_cd_by_ab[j0:j1], dtype=np.int32)
        nt_ab = int(spcd_all.size)
        if nt_ab == 0:
            continue

        A = int(spA[spab])
        B = int(spB[spab])
        a0 = int(shell_ao_start[A])
        b0 = int(shell_ao_start[B])
        nA = int(ncart(int(shell_l[A])))
        nB = int(ncart(int(shell_l[B])))
        nAB = int(nA * nB)

        CA = C_active[a0 : a0 + nA, :]
        CB = C_active[b0 : b0 + nB, :]
        K_AB = build_pair_coeff_ordered(CA, CB, same_shell=(A == B))

        B_off = cp.zeros((nAB, n_pair), dtype=cp.float64)
        B_diag = cp.zeros((nAB, n_pair), dtype=cp.float64)
        has_off = False
        has_diag = False

        task_group = TaskList(
            task_spAB=np.full((nt_ab,), np.int32(spab), dtype=np.int32),
            task_spCD=spcd_all,
        )

        for tb in iter_tile_batches_spd(
            task_group,
            shell_pairs=sp,
            shell_l=shell_l,
            dbasis=dbasis,
            dsp=dsp,
            pt=pt,
            stream=None,
            threads=int(threads),
            mode=mode,
            work_small_max=int(work_small_max),
            work_large_min=int(work_large_min),
            blocks_per_task=int(blocks_per_task),
            max_tile_bytes=int(max_tile_bytes_i),
            boys=boys,
        ):
            tile = tb.tiles
            spcd_batch = np.asarray(tb.task_spCD, dtype=np.int32)
            nt = int(spcd_batch.shape[0])
            if nt == 0:
                continue

            nCD = int(tile.shape[2])
            bytes_work_task = (int(nCD) + 2 * int(nAB)) * n_pair * 8
            max_tasks = int(max(1, max_work_bytes // max(bytes_work_task, 1)))

            for i0 in range(0, nt, max_tasks):
                i1 = min(nt, i0 + max_tasks)
                spcd = np.asarray(spcd_batch[i0:i1], dtype=np.int32)
                m = int(spcd.size)
                if m == 0:
                    continue

                tile_chunk = tile[i0:i1]
                C_shell = spA[spcd]
                D_shell = spB[spcd]
                nC = int(ncart(int(shell_l[int(C_shell[0])])))
                nD = int(ncart(int(shell_l[int(D_shell[0])])))

                c0 = sp_a0[spcd]
                d0 = sp_b0[spcd]
                ic = cp.asarray(c0, dtype=cp.int32)[:, None] + _arange(nC)[None, :]
                id_ = cp.asarray(d0, dtype=cp.int32)[:, None] + _arange(nD)[None, :]
                CC = C_active[ic, :]
                CD = C_active[id_, :]

                same_cd = C_shell == D_shell
                K_CD = _build_pair_coeff_ordered_batch(CC, CD, same_shell=same_cd)
                tmp = cp.matmul(tile_chunk, K_CD)
                tmp_sum = tmp.sum(axis=0)

                diag_idx = np.nonzero(spcd == np.int32(spab))[0]
                if int(diag_idx.size):
                    has_diag = True
                    k = int(diag_idx[0])
                    B_diag += tmp[k]
                    if m > 1:
                        has_off = True
                        B_off += tmp_sum - tmp[k]
                else:
                    has_off = True
                    B_off += tmp_sum

        if has_diag:
            _accum_KtB_ordered_blocked(out_eri_mat, K_AB, B_diag, add_transpose=False, pair_block=pair_block)
        if has_off:
            _accum_KtB_ordered_blocked(out_eri_mat, K_AB, B_off, add_transpose=True, pair_block=pair_block)

    if prof is not None and t0 is not None:
        cp.cuda.get_current_stream().synchronize()
        prof["t_eri_mat_s"] = float(time.perf_counter() - float(t0))
        prof["nao"] = int(nao)
        prof["norb"] = int(norb)
        prof["n_tasks"] = int(n_tasks_total)
        prof["threads"] = int(threads)
        prof["max_tile_bytes"] = int(max_tile_bytes_i)
        prof["algorithm"] = "ab_group"
        prof["pair_block"] = int(pair_block)


def _build_active_eri_mat_dense_rys_from_cached(*, builder, C_active, out_eri_mat, profile: dict | None = None) -> None:
    """Cached dense ERI executor selector: direct, AB-group, or auto."""

    algorithm = str(getattr(builder, "algorithm", "auto")).lower().replace("-", "_").strip()
    if algorithm == "ab_group":
        return _build_active_eri_mat_dense_rys_from_cached_ab_group(
            builder=builder,
            C_active=C_active,
            out_eri_mat=out_eri_mat,
            profile=profile,
        )
    if algorithm == "direct":
        return _build_active_eri_mat_dense_rys_from_cached_direct(
            builder=builder,
            C_active=C_active,
            out_eri_mat=out_eri_mat,
            profile=profile,
        )

    # For small active spaces, direct is typically faster because AB-group
    # setup/reduction overhead dominates.
    shape = getattr(C_active, "shape", None)
    if shape is not None and len(shape) == 2:
        norb = int(shape[1])
        n_pair = int(norb) * int(norb)
        if n_pair <= 1024:
            return _build_active_eri_mat_dense_rys_from_cached_direct(
                builder=builder,
                C_active=C_active,
                out_eri_mat=out_eri_mat,
                profile=profile,
            )

    tasks = getattr(builder, "tasks", None)
    sp = getattr(builder, "sp", None)
    ntasks = int(getattr(tasks, "ntasks", 0))
    nsp = int(np.asarray(getattr(sp, "sp_A", np.zeros((0,), dtype=np.int32))).shape[0]) if sp is not None else 0
    avg = float(ntasks) / float(max(1, nsp))
    if avg >= 3.0 and ntasks >= 256:
        return _build_active_eri_mat_dense_rys_from_cached_ab_group(
            builder=builder,
            C_active=C_active,
            out_eri_mat=out_eri_mat,
            profile=profile,
        )

    return _build_active_eri_mat_dense_rys_from_cached_direct(
        builder=builder,
        C_active=C_active,
        out_eri_mat=out_eri_mat,
        profile=profile,
    )


def _build_pu_wx_eri_mat_dense_rys_from_cached(*, builder, C_mo, C_act, out_eri_puwx, profile: dict | None = None) -> None:
    """Fast mixed-index dense executor using cached preprocessing artifacts from a builder."""

    import time
    import cupy as cp

    eps_ao_f = float(getattr(builder, "eps_ao", 0.0))

    prof = None
    t0 = None
    if profile is not None:
        profile.clear()
        prof = profile.setdefault("cueri_pu_wx_dense_rys_cached", {})
        prof["plan_cache_hit"] = True  # within-builder cache
        t0 = time.perf_counter()

    sp = getattr(builder, "sp", None)
    basis = getattr(builder, "ao_basis", None)
    class_ids = getattr(builder, "class_ids", None)
    offsets = getattr(builder, "offsets", None)
    task_ab_all = getattr(builder, "task_ab", None)
    task_cd_all = getattr(builder, "task_cd", None)
    dbasis = getattr(builder, "dbasis", None)
    dsp = getattr(builder, "dsp", None)
    pt = getattr(builder, "pair_tables", None)
    if any(x is None for x in (sp, basis, class_ids, offsets, task_ab_all, task_cd_all, dbasis, dsp, pt)):
        raise RuntimeError("builder is missing cached preprocessing/device artifacts")

    C_mo = cp.asarray(C_mo, dtype=cp.float64)
    C_act = cp.asarray(C_act, dtype=cp.float64)
    C_mo = cp.ascontiguousarray(C_mo)
    C_act = cp.ascontiguousarray(C_act)
    if C_mo.ndim != 2 or C_act.ndim != 2:
        raise ValueError("C_mo/C_act must have shape (nao, nmo)/(nao, ncas)")
    nao, nmo = map(int, C_mo.shape)
    nao2, ncas = map(int, C_act.shape)
    if nao2 != nao:
        raise ValueError("C_mo/C_act nao mismatch")

    n_pair_left = int(nmo) * int(ncas)
    n_pair_right = int(ncas) * int(ncas)
    if out_eri_puwx.shape != (n_pair_left, n_pair_right):
        raise ValueError(f"out_eri_puwx must have shape ({n_pair_left},{n_pair_right}), got {out_eri_puwx.shape}")

    out_eri_puwx.fill(0)

    shell_ao_start = np.asarray(basis.shell_ao_start, dtype=np.int64)
    spA = np.asarray(sp.sp_A, dtype=np.int32)
    spB = np.asarray(sp.sp_B, dtype=np.int32)
    sp_a0 = shell_ao_start[spA]
    sp_b0 = shell_ao_start[spB]
    sp_same = spA == spB
    shell_l = np.asarray(basis.shell_l, dtype=np.int32)

    threads = int(getattr(builder, "threads", 256))
    max_tile_bytes_i = int(getattr(builder, "max_tile_bytes", 256 << 20))
    max_work_bytes = int(max_tile_bytes_i)
    mode = str(getattr(builder, "mode", "auto")).lower().strip()
    work_small_max = int(getattr(builder, "work_small_max", 512))
    work_large_min = int(getattr(builder, "work_large_min", 200_000))
    blocks_per_task = int(getattr(builder, "blocks_per_task", 8))
    boys = str(getattr(builder, "boys", "ref")).lower().strip()

    n_tasks_total = int(getattr(getattr(builder, "tasks", None), "ntasks", int(task_ab_all.shape[0])))
    nsp = int(np.asarray(sp.sp_A, dtype=np.int32).shape[0])
    arange_cache: dict[int, cp.ndarray] = {}

    def _arange(n: int):
        arr = arange_cache.get(int(n))
        if arr is None:
            arr = cp.arange(int(n), dtype=cp.int32)
            arange_cache[int(n)] = arr
        return arr

    for g in range(int(class_ids.shape[0])):
        j0 = int(offsets[g])
        j1 = int(offsets[g + 1])
        if j1 <= j0:
            continue

        task_group = TaskList(
            task_spAB=np.asarray(task_ab_all[j0:j1], dtype=np.int32),
            task_spCD=np.asarray(task_cd_all[j0:j1], dtype=np.int32),
        )
        if task_group.ntasks == 0:
            continue

        for tb in iter_tile_batches_spd(
            task_group,
            shell_pairs=sp,
            shell_l=shell_l,
            dbasis=dbasis,
            dsp=dsp,
            pt=pt,
            stream=None,
            threads=int(threads),
            mode=mode,
            work_small_max=int(work_small_max),
            work_large_min=int(work_large_min),
            blocks_per_task=int(blocks_per_task),
            max_tile_bytes=int(max_tile_bytes_i),
            boys=boys,
        ):
            tile = tb.tiles
            spab_batch = np.asarray(tb.task_spAB, dtype=np.int32)
            spcd_batch = np.asarray(tb.task_spCD, dtype=np.int32)
            nt = int(spab_batch.shape[0])
            if nt == 0:
                continue

            nAB = int(tile.shape[1])
            nCD = int(tile.shape[2])
            bytes_work = (int(nAB) * n_pair_left + int(nCD) * n_pair_right + int(nAB) * n_pair_right) * 8
            max_tasks = int(max(1, max_work_bytes // max(bytes_work, 1)))

            for i0 in range(0, nt, max_tasks):
                i1 = min(nt, i0 + max_tasks)
                spab = spab_batch[i0:i1]
                spcd = spcd_batch[i0:i1]
                tile_chunk = tile[i0:i1]
                m = int(spab.shape[0])
                if m == 0:
                    continue

                A0 = int(spA[int(spab[0])])
                B0 = int(spB[int(spab[0])])
                C0 = int(spA[int(spcd[0])])
                D0 = int(spB[int(spcd[0])])
                nA = int(ncart(int(shell_l[A0])))
                nB = int(ncart(int(shell_l[B0])))
                nC = int(ncart(int(shell_l[C0])))
                nD = int(ncart(int(shell_l[D0])))

                a0 = sp_a0[spab]
                b0 = sp_b0[spab]
                c0 = sp_a0[spcd]
                d0 = sp_b0[spcd]

                ia = cp.asarray(a0, dtype=cp.int32)[:, None] + _arange(nA)[None, :]
                ib = cp.asarray(b0, dtype=cp.int32)[:, None] + _arange(nB)[None, :]
                ic = cp.asarray(c0, dtype=cp.int32)[:, None] + _arange(nC)[None, :]
                id_ = cp.asarray(d0, dtype=cp.int32)[:, None] + _arange(nD)[None, :]

                CA_p = C_mo[ia, :]
                CB_p = C_mo[ib, :]
                CA_u = C_act[ia, :]
                CB_u = C_act[ib, :]
                CC_u = C_act[ic, :]
                CD_u = C_act[id_, :]

                same_ab = sp_same[spab]
                same_cd = sp_same[spcd]

                K_AB_mixed = _build_pair_coeff_ordered_mixed_batch(CA_p, CB_p, CA_u, CB_u, same_shell=same_ab)
                K_CD_act = _build_pair_coeff_ordered_batch(CC_u, CD_u, same_shell=same_cd)

                tmp = cp.matmul(tile_chunk, K_CD_act)
                out_eri_puwx += K_AB_mixed.reshape((m * nAB, n_pair_left)).T @ tmp.reshape((m * nAB, n_pair_right))

                off = spab != spcd
                if bool(np.any(off)):
                    off_idx = np.nonzero(off)[0].astype(np.int32, copy=False)
                    noff = int(off_idx.size)
                    if noff:
                        off_dev = cp.asarray(off_idx, dtype=cp.int32)
                        tile_off = tile_chunk[off_dev]

                        CA_u_off = CA_u[off_dev]
                        CB_u_off = CB_u[off_dev]
                        K_AB_act_off = _build_pair_coeff_ordered_batch(
                            CA_u_off,
                            CB_u_off,
                            same_shell=sp_same[spab[off_idx]],
                        )

                        CC_u_off = CC_u[off_dev]
                        CD_u_off = CD_u[off_dev]
                        CC_p_off = C_mo[ic[off_dev], :]
                        CD_p_off = C_mo[id_[off_dev], :]
                        K_CD_mixed_off = _build_pair_coeff_ordered_mixed_batch(
                            CC_p_off,
                            CD_p_off,
                            CC_u_off,
                            CD_u_off,
                            same_shell=sp_same[spcd[off_idx]],
                        )

                        tmp2 = cp.matmul(tile_off.transpose((0, 2, 1)), K_AB_act_off)
                        out_eri_puwx += K_CD_mixed_off.reshape((noff * nCD, n_pair_left)).T @ tmp2.reshape((noff * nCD, n_pair_right))

    if profile is not None:
        try:
            cp.cuda.get_current_stream().synchronize()
        except Exception:
            pass

        profile["nao"] = int(nao)
        profile["nmo"] = int(nmo)
        profile["ncas"] = int(ncas)
        profile["eri_puwx_nbytes"] = int(getattr(out_eri_puwx, "nbytes", 0))
        profile["eps_ao"] = float(eps_ao_f)
        profile["threads"] = int(threads)
        profile["max_tile_bytes"] = int(max_tile_bytes_i)
        profile["ntasks"] = int(n_tasks_total)
        profile["nsp"] = int(nsp)

        if prof is not None and t0 is not None:
            profile["t_eri_puwx_s"] = float(time.perf_counter() - float(t0))
            prof["t_eri_puwx_s"] = float(profile["t_eri_puwx_s"])


def _build_pq_uv_eri_mat_dense_rys_from_cached(
    *, builder, C_mo, C_act, out_eri_pquv, profile: dict | None = None
) -> None:
    """Build (pq|uv) ERIs where p,q are general MO and u,v are active.

    Uses the same tiled AO integral engine as the (pu|wx) builder but with
    different MO coefficient assignments:
      - AB pair: both indices use C_mo → (nmo*nmo) ordered pairs
      - CD pair: both indices use C_act → (ncas*ncas) ordered pairs
    """

    import time
    import cupy as cp

    eps_ao_f = float(getattr(builder, "eps_ao", 0.0))

    prof = None
    t0 = None
    if profile is not None:
        profile.clear()
        prof = profile.setdefault("cueri_pq_uv_dense_rys_cached", {})
        prof["plan_cache_hit"] = True
        t0 = time.perf_counter()

    sp = getattr(builder, "sp", None)
    basis = getattr(builder, "ao_basis", None)
    class_ids = getattr(builder, "class_ids", None)
    offsets = getattr(builder, "offsets", None)
    task_ab_all = getattr(builder, "task_ab", None)
    task_cd_all = getattr(builder, "task_cd", None)
    dbasis = getattr(builder, "dbasis", None)
    dsp = getattr(builder, "dsp", None)
    pt = getattr(builder, "pair_tables", None)
    if any(x is None for x in (sp, basis, class_ids, offsets, task_ab_all, task_cd_all, dbasis, dsp, pt)):
        raise RuntimeError("builder is missing cached preprocessing/device artifacts")

    C_mo = cp.ascontiguousarray(cp.asarray(C_mo, dtype=cp.float64))
    C_act = cp.ascontiguousarray(cp.asarray(C_act, dtype=cp.float64))
    if C_mo.ndim != 2 or C_act.ndim != 2:
        raise ValueError("C_mo/C_act must have shape (nao, nmo)/(nao, ncas)")
    nao, nmo = map(int, C_mo.shape)
    nao2, ncas = map(int, C_act.shape)
    if nao2 != nao:
        raise ValueError("C_mo/C_act nao mismatch")

    n_pair_left = int(nmo) * int(nmo)
    n_pair_right = int(ncas) * int(ncas)
    if out_eri_pquv.shape != (n_pair_left, n_pair_right):
        raise ValueError(f"out shape mismatch: expected ({n_pair_left},{n_pair_right}), got {out_eri_pquv.shape}")

    out_eri_pquv.fill(0)

    shell_ao_start = np.asarray(basis.shell_ao_start, dtype=np.int64)
    spA = np.asarray(sp.sp_A, dtype=np.int32)
    spB = np.asarray(sp.sp_B, dtype=np.int32)
    sp_a0 = shell_ao_start[spA]
    sp_b0 = shell_ao_start[spB]
    sp_same = spA == spB
    shell_l = np.asarray(basis.shell_l, dtype=np.int32)

    threads = int(getattr(builder, "threads", 256))
    max_tile_bytes_i = int(getattr(builder, "max_tile_bytes", 256 << 20))
    max_work_bytes = int(max_tile_bytes_i)
    mode = str(getattr(builder, "mode", "auto")).lower().strip()
    work_small_max = int(getattr(builder, "work_small_max", 512))
    work_large_min = int(getattr(builder, "work_large_min", 200_000))
    blocks_per_task = int(getattr(builder, "blocks_per_task", 8))
    boys = str(getattr(builder, "boys", "ref")).lower().strip()

    n_tasks_total = int(getattr(getattr(builder, "tasks", None), "ntasks", int(task_ab_all.shape[0])))
    arange_cache: dict[int, cp.ndarray] = {}

    def _arange(n: int):
        arr = arange_cache.get(int(n))
        if arr is None:
            arr = cp.arange(int(n), dtype=cp.int32)
            arange_cache[int(n)] = arr
        return arr

    for g in range(int(class_ids.shape[0])):
        j0 = int(offsets[g])
        j1 = int(offsets[g + 1])
        if j1 <= j0:
            continue

        task_group = TaskList(
            task_spAB=np.asarray(task_ab_all[j0:j1], dtype=np.int32),
            task_spCD=np.asarray(task_cd_all[j0:j1], dtype=np.int32),
        )
        if task_group.ntasks == 0:
            continue

        for tb in iter_tile_batches_spd(
            task_group, shell_pairs=sp, shell_l=shell_l,
            dbasis=dbasis, dsp=dsp, pt=pt, stream=None,
            threads=int(threads), mode=mode,
            work_small_max=int(work_small_max),
            work_large_min=int(work_large_min),
            blocks_per_task=int(blocks_per_task),
            max_tile_bytes=int(max_tile_bytes_i), boys=boys,
        ):
            tile = tb.tiles
            spab_batch = np.asarray(tb.task_spAB, dtype=np.int32)
            spcd_batch = np.asarray(tb.task_spCD, dtype=np.int32)
            nt = int(spab_batch.shape[0])
            if nt == 0:
                continue

            nAB = int(tile.shape[1])
            nCD = int(tile.shape[2])
            bytes_work = (int(nAB) * n_pair_left + int(nCD) * n_pair_right + int(nAB) * n_pair_right) * 8
            max_tasks = int(max(1, max_work_bytes // max(bytes_work, 1)))

            for i0 in range(0, nt, max_tasks):
                i1 = min(nt, i0 + max_tasks)
                spab = spab_batch[i0:i1]
                spcd = spcd_batch[i0:i1]
                tile_chunk = tile[i0:i1]
                m = int(spab.shape[0])
                if m == 0:
                    continue

                A0 = int(spA[int(spab[0])])
                B0 = int(spB[int(spab[0])])
                C0 = int(spA[int(spcd[0])])
                D0 = int(spB[int(spcd[0])])

                ia = cp.asarray(sp_a0[spab], dtype=cp.int32)[:, None] + _arange(ncart(int(shell_l[A0])))[None, :]
                ib = cp.asarray(sp_b0[spab], dtype=cp.int32)[:, None] + _arange(ncart(int(shell_l[B0])))[None, :]
                ic = cp.asarray(sp_a0[spcd], dtype=cp.int32)[:, None] + _arange(ncart(int(shell_l[C0])))[None, :]
                id_ = cp.asarray(sp_b0[spcd], dtype=cp.int32)[:, None] + _arange(ncart(int(shell_l[D0])))[None, :]

                # AB pair: both use C_mo
                CA_p = C_mo[ia, :]
                CB_p = C_mo[ib, :]
                K_AB_mo = _build_pair_coeff_ordered_batch(CA_p, CB_p, same_shell=sp_same[spab])

                # CD pair: both use C_act
                CC_u = C_act[ic, :]
                CD_u = C_act[id_, :]
                K_CD_act = _build_pair_coeff_ordered_batch(CC_u, CD_u, same_shell=sp_same[spcd])

                tmp = cp.matmul(tile_chunk, K_CD_act)
                out_eri_pquv += K_AB_mo.reshape((m * nAB, n_pair_left)).T @ tmp.reshape((m * nAB, n_pair_right))

                off = spab != spcd
                if bool(np.any(off)):
                    off_idx = np.nonzero(off)[0].astype(np.int32, copy=False)
                    noff = int(off_idx.size)
                    if noff:
                        off_dev = cp.asarray(off_idx, dtype=cp.int32)
                        tile_off = tile_chunk[off_dev]

                        # Swapped: AB uses C_act, CD uses C_mo
                        CA_u_off = C_act[ia[off_dev], :]
                        CB_u_off = C_act[ib[off_dev], :]
                        K_AB_act_off = _build_pair_coeff_ordered_batch(CA_u_off, CB_u_off, same_shell=sp_same[spab[off_idx]])

                        CC_p_off = C_mo[ic[off_dev], :]
                        CD_p_off = C_mo[id_[off_dev], :]
                        K_CD_mo_off = _build_pair_coeff_ordered_batch(CC_p_off, CD_p_off, same_shell=sp_same[spcd[off_idx]])

                        tmp2 = cp.matmul(tile_off.transpose((0, 2, 1)), K_AB_act_off)
                        out_eri_pquv += K_CD_mo_off.reshape((noff * nCD, n_pair_left)).T @ tmp2.reshape((noff * nCD, n_pair_right))

    if profile is not None:
        try:
            cp.cuda.get_current_stream().synchronize()
        except Exception:
            pass
        profile["nao"] = int(nao)
        profile["nmo"] = int(nmo)
        profile["ncas"] = int(ncas)
        if prof is not None and t0 is not None:
            import time as _time
            profile["t_eri_pquv_s"] = float(_time.perf_counter() - float(t0))
            prof["t_eri_pquv_s"] = float(profile["t_eri_pquv_s"])


@with_stream
def build_active_eri_mat_dense_rys(
    ao_basis,
    C_active,
    *,
    stream=None,
    threads: int = 256,
    max_tile_bytes: int = 256 * 1024 * 1024,
    eps_ao: float = 0.0,
    sp_Q=None,
):
    """Compute active-space ERIs as a dense ordered-pair matrix (pq|rs) on GPU.

    This entrypoint now resolves a cached `CuERIActiveSpaceDenseGPUBuilder` and
    executes its cached dense builder path, avoiding the old per-task Python loop.
    """

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for GPU dense build") from e

    eps_ao_f = float(eps_ao)
    if eps_ao_f < 0.0:
        raise ValueError("eps_ao must be >= 0")
    max_tile_bytes_i = int(max_tile_bytes)
    if max_tile_bytes_i <= 0:
        raise ValueError("max_tile_bytes must be > 0")
    builder = _resolve_active_dense_gpu_builder(
        ao_basis,
        threads=int(threads),
        max_tile_bytes=int(max_tile_bytes_i),
        eps_ao=float(eps_ao_f),
        sp_Q=sp_Q,
        algorithm="auto",
        mode="auto",
        work_small_max=512,
        work_large_min=200_000,
        blocks_per_task=8,
        boys="ref",
    )
    return builder.build(C_active).eri_mat


@with_stream
def build_pu_wx_eri_mat_dense_rys(
    ao_basis,
    C_mo,
    C_act,
    *,
    stream=None,
    threads: int = 256,
    max_tile_bytes: int = 256 * 1024 * 1024,
    eps_ao: float = 0.0,
    sp_Q=None,
    profile: dict | None = None,
    builder=None,
):
    """Compute mixed-index ERIs (pu|wx) as a dense ordered-pair matrix.

    Evaluates integrals of the form (pu|wx) where p is a general MO, u is an active orbital,
    and w, x are active orbitals. Used for active-space 2-RDM gradient terms.

    Parameters
    ----------
    ao_basis : BasisCartSoA | object
        The atomic orbital basis set definition.
    C_mo : np.ndarray | cupy.ndarray
        General MO coefficients (index p). Shape: `(nao, nmo)`.
    C_act : np.ndarray | cupy.ndarray
        Active MO coefficients (indices u, w, x). Shape: `(nao, ncas)`.
    stream : cupy.cuda.Stream | None, optional
        CUDA stream for asynchronous execution.
    threads : int, default=256
        Block size for CUDA kernels.
    max_tile_bytes : int, default=268435456
        Maximum GPU memory for temporary buffers.
    eps_ao : float, default=0.0
        AO screening threshold.
    sp_Q : np.ndarray | None, optional
        Pre-computed Schwarz screening values.
    profile : dict | None, optional
        Dictionary for collecting performance metrics.
    builder : CuERIActiveSpaceDenseGPUBuilder | None, optional
        Existing builder instance to reuse cached data.

    Returns
    -------
    cupy.ndarray
        The ERI matrix of shape `(nmo*ncas, ncas*ncas)`.
        Row index `pu = p * ncas + u`, Col index `wx = w * ncas + x`.

    Notes
    -----
    - Uses the generic Rys quadrature backend.
    """

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for GPU dense build") from e

    eps_ao_f = float(eps_ao)
    if eps_ao_f < 0.0:
        raise ValueError("eps_ao must be >= 0")
    max_tile_bytes_i = int(max_tile_bytes)
    if max_tile_bytes_i <= 0:
        raise ValueError("max_tile_bytes must be > 0")

    builder_cache_hit = False

    # Resolve a cached/prebuilt builder if not provided.
    if builder is None:
        try:
            from .active_space_dense_gpu import CuERIActiveSpaceDenseGPUBuilder  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cuERI is required for GPU dense build") from e

        dev = int(cp.cuda.runtime.getDevice())
        spq_id = 0 if sp_Q is None else int(id(sp_Q))
        key = (dev, int(id(ao_basis)), int(threads), int(max_tile_bytes_i), float(eps_ao_f), int(spq_id))
        hit = _dense_gpu_puwx_builder_cache.get(key)
        if hit is not None:
            basis_ref, bld = hit
            if basis_ref() is ao_basis:
                builder = bld
                builder_cache_hit = True
            else:
                del _dense_gpu_puwx_builder_cache[key]

        if builder is None:
            builder = CuERIActiveSpaceDenseGPUBuilder(
                ao_basis=ao_basis,
                threads=int(threads),
                max_tile_bytes=int(max_tile_bytes_i),
                eps_ao=float(eps_ao_f),
                sp_Q=None if sp_Q is None else np.asarray(sp_Q, dtype=np.float64),
                stream=None,
            )
            _dense_gpu_puwx_builder_cache[key] = (weakref.ref(ao_basis), builder)
            while len(_dense_gpu_puwx_builder_cache) > _DENSE_GPU_PUWX_BUILDER_CACHE_MAX:
                _dense_gpu_puwx_builder_cache.pop(next(iter(_dense_gpu_puwx_builder_cache)))

    # Execute via the cached builder.
    out = builder.build_pu_wx_eri_mat(C_mo, C_act, profile=profile)
    if profile is not None:
        profile["builder_cache_hit"] = bool(builder_cache_hit)
    return out


@with_stream
def build_active_eri_mat_dense_sp_only(
    ao_basis,
    C_active,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "auto",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    """Compute active-space ERIs as an ordered-pair matrix.

    This entrypoint preserves the historical `*_sp_only` API but now routes to the
    cached general dense builder path for consistency and performance.

    Parameters
    ----------
    ao_basis : BasisCartSoA | object
        The atomic orbital basis set definition.
    C_active : np.ndarray | cupy.ndarray
        Active space MO coefficients. Shape: `(nao, norb)`.
    stream : cupy.cuda.Stream | None, optional
        CUDA stream for asynchronous execution.
    threads : int, default=256
        Block size for CUDA kernels.
    mode : str, default='auto'
        Execution mode for the GPU kernels ('auto', 'block', 'warp', 'multiblock').
    work_small_max : int, default=512
        Threshold for small workload kernel dispatch.
    work_large_min : int, default=200000
        Threshold for large workload kernel dispatch.
    blocks_per_task : int, default=8
        Number of thread blocks per task for certain modes.

    Returns
    -------
    cupy.ndarray
        The ERI matrix of shape `(norb*norb, norb*norb)`.
        Row index `pq = p * norb + q`, Col index `rs = r * norb + s`.

    Notes
    -----
    - In current implementation, this function delegates to the cached dense builder
      and supports higher angular momentum shells (e.g. triple-zeta bases with f shells).
    """

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for GPU dense build") from e

    C_active = cp.asarray(C_active, dtype=cp.float64)
    C_active = cp.ascontiguousarray(C_active)
    if C_active.ndim != 2:
        raise ValueError("C_active must have shape (nao, norb)")

    builder = _resolve_active_dense_gpu_builder(
        ao_basis,
        threads=int(threads),
        max_tile_bytes=int(256 << 20),
        eps_ao=0.0,
        sp_Q=None,
        algorithm="auto",
        mode=str(mode),
        work_small_max=int(work_small_max),
        work_large_min=int(work_large_min),
        blocks_per_task=int(blocks_per_task),
        boys="ref",
    )
    return builder.build(C_active).eri_mat


@with_stream
def build_active_eri_packed_dense_sp_only(
    ao_basis,
    C_active,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "auto",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
    max_tile_bytes: int = 256 * 1024 * 1024,
    eps_ao: float = 0.0,
    eps_mo: float = 0.0,
    sp_Q=None,
    algorithm: str = "direct",
):
    """Compute active-space ERIs as a packed pair matrix on GPU.

    Uses the cached general dense builder for all angular momenta. For `eps_mo>0`,
    additional shell-pair screening is applied before dense accumulation.
    """

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for GPU dense build") from e

    C_active = cp.asarray(C_active, dtype=cp.float64)
    C_active = cp.ascontiguousarray(C_active)
    if C_active.ndim != 2:
        raise ValueError("C_active must have shape (nao, norb)")
    _nao, norb = map(int, C_active.shape)
    if _nao <= 0 or norb <= 0:
        raise ValueError("C_active must be non-empty (nao>0 and norb>0)")

    if eps_ao < 0.0 or eps_mo < 0.0:
        raise ValueError("eps_ao/eps_mo must be >= 0")

    max_tile_bytes_i = int(max_tile_bytes)
    if max_tile_bytes_i <= 0:
        raise ValueError("max_tile_bytes must be > 0")

    mode = mode.lower().strip()
    if mode not in ("block", "warp", "multiblock", "auto"):
        raise ValueError("mode must be one of: 'block', 'warp', 'multiblock', 'auto'")

    algorithm = algorithm.lower().strip()
    if algorithm not in ("direct", "ab_group"):
        raise ValueError("algorithm must be one of: 'direct', 'ab_group'")

    builder = _resolve_active_dense_gpu_builder(
        ao_basis,
        threads=int(threads),
        max_tile_bytes=int(max_tile_bytes_i),
        eps_ao=float(eps_ao),
        sp_Q=sp_Q,
        algorithm=str(algorithm),
        mode=str(mode),
        work_small_max=int(work_small_max),
        work_large_min=int(work_large_min),
        blocks_per_task=int(blocks_per_task),
        boys="ref",
    )

    if float(eps_mo) > 0.0:
        b_exec = _filter_builder_tasks_by_eps_mo(builder, C_active, eps_mo=float(eps_mo))
        n_pair = int(norb) * int(norb)
        eri_ordered = cp.zeros((n_pair, n_pair), dtype=cp.float64)
        _build_active_eri_mat_dense_rys_from_cached(
            builder=b_exec,
            C_active=C_active,
            out_eri_mat=eri_ordered,
            profile=None,
        )
    else:
        eri_ordered = builder.build(C_active).eri_mat

    tri_p, tri_q = cp.tril_indices(int(norb))
    ord_idx = tri_p * int(norb) + tri_q
    return eri_ordered[ord_idx[:, None], ord_idx[None, :]]


__all__ = [
    "build_active_eri_mat_dense_rys",
    "build_active_eri_mat_dense_sp_only",
    "build_active_eri_packed_dense_sp_only",
    "build_pu_wx_eri_mat_dense_rys",
]
