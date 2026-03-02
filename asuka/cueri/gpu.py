from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import weakref

from .basis import BasisSoA
from .cart import ncart
from .shell_pairs import ShellPairs
from .shell_pairs import build_shell_pairs
from .shell_pairs import build_shell_pairs_l_order
from .stream import stream_ctx, stream_ptr
from .tasks import TaskList
from .tasks import decode_eri_class_id, eri_class_id, group_tasks_by_class, with_task_class_id

try:  # optional CUDA extension
    from . import _cueri_cuda_ext as _ext
except Exception:  # pragma: no cover
    _ext = None


def has_cuda_ext() -> bool:
    return _ext is not None


def _detect_cuda_kernel_limits(*, default_lmax: int = 5, default_nroots: int = 11) -> tuple[int, int]:
    """Best-effort discovery of compiled CUDA ERI limits from the extension."""

    lmax = int(default_lmax)
    nroots = int(default_nroots)
    if _ext is None:
        return lmax, nroots
    limits_fn = getattr(_ext, "kernel_limits_device", None)
    if limits_fn is None:
        return lmax, nroots
    cp = None
    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None
    try:
        limits = dict(limits_fn())
        lmax_i = int(limits.get("lmax", lmax))
        nroots_i = int(limits.get("nroots_max", nroots))
    except Exception:
        # If the extension probe tripped a CUDA runtime error during import-time detection,
        # clear the sticky runtime status so later availability checks remain reliable.
        if cp is not None:
            try:
                cp.cuda.runtime.getLastError()
            except Exception:
                pass
        return lmax, nroots
    finally:
        if cp is not None:
            try:
                cp.cuda.runtime.getLastError()
            except Exception:
                pass
    if lmax_i < 0 or nroots_i < 1:
        return lmax, nroots
    return lmax_i, nroots_i


CUDA_MAX_L, CUDA_MAX_NROOTS = _detect_cuda_kernel_limits()


def warmup_cuda(*, gemm_n: int = 256, chol_n: int = 256, stream=None) -> None:
    """Warm up CUDA libraries (cuBLAS/cuSOLVER) to reduce first-call overhead.

    Motivation: the first call into cuBLAS/cuSOLVER-backed routines (e.g. GEMM,
    Cholesky, triangular solves) can pay a noticeable one-time initialization
    cost. cuERI DF builds use these ops (e.g. metric Cholesky + triangular
    solves), so warming up once at program start can substantially reduce the
    first DF build wall time.
    """

    try:
        import cupy as cp
        import cupyx.scipy.linalg as cpx_linalg
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for warmup_cuda") from e

    gemm_n = int(gemm_n)
    chol_n = int(chol_n)
    if gemm_n <= 0 or chol_n <= 0:
        raise ValueError("gemm_n/chol_n must be > 0")

    with stream_ctx(stream):
        s = cp.cuda.get_current_stream()
        A = cp.random.standard_normal((gemm_n, gemm_n), dtype=cp.float64)
        _ = A @ A.T

        B = cp.random.standard_normal((chol_n, chol_n), dtype=cp.float64)
        SPD = B @ B.T + 1e-3 * cp.eye(chol_n, dtype=cp.float64)
        L = cp.linalg.cholesky(SPD)
        _ = cpx_linalg.solve_triangular(L, B, lower=True, trans="N", unit_diagonal=False, overwrite_b=False)
        s.synchronize()


@dataclass(frozen=True)
class DeviceBasisSS:
    shell_cx: object
    shell_cy: object
    shell_cz: object
    shell_prim_start: object
    shell_nprim: object
    prim_exp: object
    prim_coef: object


@dataclass(frozen=True)
class DeviceShellPairs:
    sp_A: object
    sp_B: object
    sp_npair: object
    sp_pair_start: object


@dataclass(frozen=True)
class DevicePairTables:
    pair_eta: object
    pair_Px: object
    pair_Py: object
    pair_Pz: object
    pair_cK: object


@dataclass(frozen=True)
class _DFMetric2c2eRysPlan:
    dbasis: DeviceBasisSS
    dsp: DeviceShellPairs
    pt: DevicePairTables
    batches: list[object]
    batch_p0_dev: list[object]
    batch_q0_dev: list[object]
    naux: int


@dataclass(frozen=True)
class _DFInt3c2eRysPlan:
    dbasis: DeviceBasisSS
    dsp: DeviceShellPairs
    pt: DevicePairTables
    batches: list[object]
    batch_a0_dev: list[object]
    batch_b0_dev: list[object]
    batch_a0_sph_dev: list[object]
    batch_b0_sph_dev: list[object]
    batch_p0_dev: list[object]
    batch_nB: list[int]
    batch_la: list[int]
    batch_lb: list[int]
    nao: int
    nao_sph: int
    naux: int


_DF_PLAN_CACHE_MAX = 4
_df_metric_2c2e_rys_plan_cache: dict[tuple, tuple[weakref.ref, _DFMetric2c2eRysPlan]] = {}
_df_int3c2e_rys_plan_cache: dict[tuple, tuple[weakref.ref, weakref.ref, _DFInt3c2eRysPlan]] = {}


@dataclass(frozen=True)
class _DFInt3c2eRysContractedAOPlan:
    dbasis: DeviceBasisSS
    dsp: DeviceShellPairs
    pt: DevicePairTables
    tasks: TaskList
    perm: np.ndarray  # int32, grouped by task_class_id
    class_ids: np.ndarray  # int32, unique class ids
    offsets: np.ndarray  # int32, group offsets (len=ncid+1)

    ao_shell_ao_start_dev: object
    ao_shell_nctr_dev: object
    ao_shell_coef_start_dev: object
    ao_prim_coef_dev: object
    aux_shell_ao_start_dev: object

    n_shell_ao: int
    nao: int
    naux: int


_df_int3c2e_rys_contracted_ao_plan_cache: dict[tuple, tuple[weakref.ref, weakref.ref, _DFInt3c2eRysContractedAOPlan]] = {}

# Cache for an "expanded contractions" view (BasisCartSoA) of contracted AO bases.
#
# This is used by the contracted-AO hybrid DF path to reuse existing Step-2 kernels
# for Step-2-eligible task classes. The expanded view must be cached to make
# `_get_df_int3c2e_rys_plan` cache effective (it keys by id(ao_basis)).
_expanded_cart_basis_cache: dict[int, tuple[weakref.ref, object]] = {}

# Kernel classes with native Step-2 kernels (see `cueri/eri_dispatch.py`).
_STEP2_BASE_CLASS_IDS: set[int] = {
    int(eri_class_id(0, 0, 0, 0)),  # ssss
    int(eri_class_id(1, 0, 0, 0)),  # psss
    int(eri_class_id(1, 1, 0, 0)),  # ppss
    int(eri_class_id(1, 0, 1, 0)),  # psps
    int(eri_class_id(1, 1, 1, 0)),  # ppps
    int(eri_class_id(1, 1, 1, 1)),  # pppp
    int(eri_class_id(2, 0, 0, 0)),  # dsss
    int(eri_class_id(2, 2, 0, 0)),  # ddss
    int(eri_class_id(0, 0, 2, 1)),  # ssdp
    int(eri_class_id(1, 0, 2, 0)),  # psds
    int(eri_class_id(1, 0, 2, 1)),  # psdp
    int(eri_class_id(1, 0, 2, 2)),  # psdd
    int(eri_class_id(1, 1, 2, 0)),  # ppds
    int(eri_class_id(1, 1, 2, 1)),  # ppdp
    int(eri_class_id(1, 1, 2, 2)),  # ppdd
    int(eri_class_id(2, 0, 2, 0)),  # dsds
    int(eri_class_id(2, 0, 2, 1)),  # dsdp
    int(eri_class_id(2, 0, 2, 2)),  # dsdd
    int(eri_class_id(3, 1, 0, 0)),  # fpss
    int(eri_class_id(3, 2, 0, 0)),  # fdss
    int(eri_class_id(3, 3, 0, 0)),  # ffss
    int(eri_class_id(3, 1, 1, 0)),  # fpps
    int(eri_class_id(3, 2, 1, 0)),  # fdps
    int(eri_class_id(3, 3, 1, 0)),  # ffps
    int(eri_class_id(3, 1, 2, 0)),  # fpds
    int(eri_class_id(3, 2, 2, 0)),  # fdds
    int(eri_class_id(3, 3, 2, 0)),  # ffds
    int(eri_class_id(0, 0, 3, 0)),  # ssfs
    int(eri_class_id(1, 0, 3, 0)),  # psfs
    int(eri_class_id(1, 1, 3, 0)),  # ppfs
    int(eri_class_id(2, 0, 3, 0)),  # dsfs
    int(eri_class_id(3, 0, 3, 0)),  # fsfs
    int(eri_class_id(2, 1, 3, 0)),  # dpfs
    int(eri_class_id(3, 1, 3, 0)),  # fpfs
    int(eri_class_id(2, 2, 3, 0)),  # ddfs
    int(eri_class_id(3, 2, 3, 0)),  # fdfs
    int(eri_class_id(3, 3, 3, 0)),  # fffs
    int(eri_class_id(0, 0, 4, 0)),  # ssgs
    int(eri_class_id(1, 0, 4, 0)),  # psgs
    int(eri_class_id(1, 1, 4, 0)),  # ppgs
    int(eri_class_id(2, 0, 4, 0)),  # dsgs
    int(eri_class_id(3, 0, 4, 0)),  # fsgs
    int(eri_class_id(2, 1, 4, 0)),  # dpgs
    int(eri_class_id(3, 1, 4, 0)),  # fpgs
    int(eri_class_id(2, 2, 4, 0)),  # ddgs
    int(eri_class_id(3, 2, 4, 0)),  # fdgs
    int(eri_class_id(3, 3, 4, 0)),  # ffgs
    int(eri_class_id(2, 1, 2, 1)),  # dpdp
    int(eri_class_id(2, 1, 2, 2)),  # dpdd
    int(eri_class_id(2, 2, 2, 2)),  # dddd
}


def _trim_plan_cache(cache: dict, max_items: int = _DF_PLAN_CACHE_MAX) -> None:
    while len(cache) > int(max_items):
        cache.pop(next(iter(cache)))


def _get_expanded_cart_basis_from_contracted(ao_basis):
    key = int(id(ao_basis))
    hit = _expanded_cart_basis_cache.get(key)
    if hit is not None:
        ao_ref, ao_exp = hit
        if ao_ref() is ao_basis:
            return ao_exp
        del _expanded_cart_basis_cache[key]

    from .basis_utils import expand_contracted_cart_basis

    ao_exp = expand_contracted_cart_basis(ao_basis)
    _expanded_cart_basis_cache[key] = (weakref.ref(ao_basis), ao_exp)
    _trim_plan_cache(_expanded_cart_basis_cache)
    return ao_exp


def _df_int3c2e_class_id_step2_eligible(cid: int) -> bool:
    """Return True if a DF (la,lb,lc,ld) class can be handled by a Step-2 kernel (possibly via swap+transpose)."""

    cid_u32 = int(cid) & 0xFFFFFFFF
    if cid_u32 in _STEP2_BASE_CLASS_IDS:
        return True
    la = cid_u32 & 0xFF
    lb = (cid_u32 >> 8) & 0xFF
    lc = (cid_u32 >> 16) & 0xFF
    ld = (cid_u32 >> 24) & 0xFF
    swap_cid = int((lc | (ld << 8) | (la << 16) | (lb << 24)) & 0xFFFFFFFF)
    return swap_cid in _STEP2_BASE_CLASS_IDS


def _basis_with_dummy_constant_shell(basis: BasisSoA) -> tuple[BasisSoA, int]:
    """Append a single 'constant' shell with one primitive exp=0, coef=1.

    This is used to reuse the (ss|ss) evaluator to represent:
      - 2c2e metric: (P|Q) == (P*1 | Q*1)
      - 3c2e integrals: (μν|P) == (μν | P*1)
    """

    basis = BasisSoA(
        shell_cxyz=np.asarray(basis.shell_cxyz, dtype=np.float64, order="C"),
        shell_prim_start=np.asarray(basis.shell_prim_start, dtype=np.int32, order="C"),
        shell_nprim=np.asarray(basis.shell_nprim, dtype=np.int32, order="C"),
        shell_l=np.asarray(basis.shell_l, dtype=np.int32, order="C"),
        prim_exp=np.asarray(basis.prim_exp, dtype=np.float64, order="C"),
        prim_coef=np.asarray(basis.prim_coef, dtype=np.float64, order="C"),
        source_bas_id=basis.source_bas_id,
        source_ctr_id=basis.source_ctr_id,
    )

    n_shell = int(basis.shell_cxyz.shape[0])
    n_prim = int(basis.prim_exp.shape[0])

    shell_cxyz = np.concatenate([basis.shell_cxyz, np.zeros((1, 3), dtype=np.float64)], axis=0)
    shell_prim_start = np.concatenate([basis.shell_prim_start, np.asarray([n_prim], dtype=np.int32)])
    shell_nprim = np.concatenate([basis.shell_nprim, np.asarray([1], dtype=np.int32)])
    shell_l = np.concatenate([basis.shell_l, np.asarray([0], dtype=np.int32)])
    prim_exp = np.concatenate([basis.prim_exp, np.asarray([0.0], dtype=np.float64)])
    prim_coef = np.concatenate([basis.prim_coef, np.asarray([1.0], dtype=np.float64)])

    bas_id = basis.source_bas_id
    ctr_id = basis.source_ctr_id
    if bas_id is not None:
        bas_id = np.concatenate([np.asarray(bas_id, dtype=np.int32), np.asarray([-1], dtype=np.int32)])
    if ctr_id is not None:
        ctr_id = np.concatenate([np.asarray(ctr_id, dtype=np.int32), np.asarray([-1], dtype=np.int32)])

    out = BasisSoA(
        shell_cxyz=shell_cxyz,
        shell_prim_start=shell_prim_start,
        shell_nprim=shell_nprim,
        shell_l=shell_l,
        prim_exp=prim_exp,
        prim_coef=prim_coef,
        source_bas_id=bas_id,
        source_ctr_id=ctr_id,
    )
    return out, n_shell


def _shell_pairs_dummy_times_real(basis_with_dummy: BasisSoA, dummy_shell: int) -> ShellPairs:
    """ShellPairs for sp=(dummy, P) for each real shell P (P < dummy)."""

    n_shell = int(basis_with_dummy.shell_cxyz.shape[0])
    if not (0 <= dummy_shell < n_shell):
        raise ValueError("dummy_shell out of range")

    n_real = int(dummy_shell)
    sp_A = np.full((n_real,), int(dummy_shell), dtype=np.int32)
    sp_B = np.arange(n_real, dtype=np.int32)
    sp_npair = np.asarray(basis_with_dummy.shell_nprim[:n_real], dtype=np.int32)
    sp_pair_start = np.empty((n_real + 1,), dtype=np.int32)
    sp_pair_start[0] = 0
    sp_pair_start[1:] = np.cumsum(sp_npair, dtype=np.int32)
    return ShellPairs(sp_A=sp_A, sp_B=sp_B, sp_npair=sp_npair, sp_pair_start=sp_pair_start)


def _require_cuda_ext() -> None:
    if _ext is None:
        raise RuntimeError("cuERI CUDA extension not available; build via `python -m asuka.cueri.build_cuda_ext`")


def _stream_ptr(stream) -> int:
    return stream_ptr(stream)


def sph_coeff_sph_to_cart_device(
    C_sph,
    *,
    ao2shell_cart,
    ao2local_cart,
    shell_ao_start_sph,
    shell_l,
    out=None,
    stream=None,
    threads: int = 256,
):
    """Transform spherical AO coefficients to the cartesian AO layout used by cuERI kernels."""

    import cupy as cp

    _require_cuda_ext()
    with stream_ctx(stream):
        C_sph = cp.asarray(C_sph, dtype=cp.float64)
        C_sph = cp.ascontiguousarray(C_sph)
        if C_sph.ndim != 2:
            raise ValueError("C_sph must have shape (nao_sph, norb)")
        _nao_sph, norb = map(int, C_sph.shape)

        ao2shell_cart = cp.asarray(ao2shell_cart, dtype=cp.int32)
        ao2local_cart = cp.asarray(ao2local_cart, dtype=cp.int32)
        shell_ao_start_sph = cp.asarray(shell_ao_start_sph, dtype=cp.int32)
        shell_l = cp.asarray(shell_l, dtype=cp.int32)

        ao2shell_cart = cp.ascontiguousarray(ao2shell_cart)
        ao2local_cart = cp.ascontiguousarray(ao2local_cart)
        shell_ao_start_sph = cp.ascontiguousarray(shell_ao_start_sph)
        shell_l = cp.ascontiguousarray(shell_l)

        if ao2shell_cart.ndim != 1 or ao2local_cart.ndim != 1:
            raise ValueError("ao2shell_cart/ao2local_cart must be 1D arrays")
        if shell_ao_start_sph.ndim != 1 or shell_l.ndim != 1:
            raise ValueError("shell_ao_start_sph/shell_l must be 1D arrays")
        nao_cart = int(ao2shell_cart.shape[0])
        if int(ao2local_cart.shape[0]) != nao_cart:
            raise ValueError("ao2shell_cart/ao2local_cart length mismatch")
        if int(shell_ao_start_sph.shape[0]) != int(shell_l.shape[0]):
            raise ValueError("shell_ao_start_sph/shell_l length mismatch")

        if out is None:
            out = cp.empty((nao_cart, norb), dtype=cp.float64)
        else:
            out = cp.asarray(out, dtype=cp.float64)
            if tuple(out.shape) != (nao_cart, norb):
                raise ValueError(f"out must have shape ({nao_cart}, {norb}), got {tuple(out.shape)}")
            out = cp.ascontiguousarray(out)

        _ext.sph_coeff_sph_to_cart_device(
            C_sph,
            out,
            ao2shell_cart,
            ao2local_cart,
            shell_ao_start_sph,
            shell_l,
            int(threads),
            int(_stream_ptr(stream)),
            False,
        )
        return out


def cart2sph_eri_tiles_device(
    tile_cart,
    *,
    la: int,
    lb: int,
    lc: int,
    ld: int,
    out=None,
    tmp=None,
    stream=None,
    threads: int = 256,
):
    """Transform a batch of contracted ERI tiles from Cartesian to spherical AOs on the GPU.

    Parameters
    ----------
    tile_cart
        CuPy array with shape (ntasks, nAB_cart, nCD_cart) and dtype float64.
    la, lb, lc, ld
        Angular momenta for the shell quartet class corresponding to the tile axes.
    out
        Optional preallocated output array with shape (ntasks, nAB_sph, nCD_sph).
    tmp
        Optional preallocated scratch array with shape (ntasks, nAB_cart, nCD_sph).

    Returns
    -------
    cupy.ndarray
        Spherical tile with shape (ntasks, nAB_sph, nCD_sph).
    """

    import cupy as cp

    _require_cuda_ext()
    la = int(la)
    lb = int(lb)
    lc = int(lc)
    ld = int(ld)
    with stream_ctx(stream):
        tile_cart = cp.asarray(tile_cart, dtype=cp.float64)
        tile_cart = cp.ascontiguousarray(tile_cart)
        if tile_cart.ndim != 3:
            raise ValueError("tile_cart must have shape (ntasks, nAB, nCD)")
        ntasks = int(tile_cart.shape[0])

        nA_cart = int(ncart(la))
        nB_cart = int(ncart(lb))
        nC_cart = int(ncart(lc))
        nD_cart = int(ncart(ld))
        nAB_cart = nA_cart * nB_cart
        nCD_cart = nC_cart * nD_cart
        if int(tile_cart.shape[1]) != nAB_cart or int(tile_cart.shape[2]) != nCD_cart:
            raise ValueError(
                f"tile_cart has shape {tuple(tile_cart.shape)}, expected (nt, {nAB_cart}, {nCD_cart}) for (la,lb,lc,ld)=({la},{lb},{lc},{ld})"
            )

        nA_sph = 2 * la + 1
        nB_sph = 2 * lb + 1
        nC_sph = 2 * lc + 1
        nD_sph = 2 * ld + 1
        nAB_sph = nA_sph * nB_sph
        nCD_sph = nC_sph * nD_sph

        if tmp is None:
            tmp = cp.empty((ntasks, nAB_cart, nCD_sph), dtype=cp.float64)
        else:
            tmp = cp.asarray(tmp, dtype=cp.float64)
            if tuple(tmp.shape) != (ntasks, nAB_cart, nCD_sph):
                raise ValueError(f"tmp must have shape {(ntasks, nAB_cart, nCD_sph)}, got {tuple(tmp.shape)}")
            tmp = cp.ascontiguousarray(tmp)

        if out is None:
            out = cp.empty((ntasks, nAB_sph, nCD_sph), dtype=cp.float64)
        else:
            out = cp.asarray(out, dtype=cp.float64)
            if tuple(out.shape) != (ntasks, nAB_sph, nCD_sph):
                raise ValueError(f"out must have shape {(ntasks, nAB_sph, nCD_sph)}, got {tuple(out.shape)}")
            out = cp.ascontiguousarray(out)

        _ext.cart2sph_eri_right_device(
            tile_cart.ravel(),
            tmp.ravel(),
            int(la),
            int(lb),
            int(lc),
            int(ld),
            int(threads),
            int(_stream_ptr(stream)),
            False,
        )
        _ext.cart2sph_eri_left_device(
            tmp.ravel(),
            out.ravel(),
            int(la),
            int(lb),
            int(lc),
            int(ld),
            int(threads),
            int(_stream_ptr(stream)),
            False,
        )
        return out


def scatter_eri_tiles_sph_s8_inplace_device(
    tasks: TaskList,
    shell_pairs: DeviceShellPairs,
    *,
    shell_ao_start_sph,
    nao_sph: int,
    nA: int,
    nB: int,
    nC: int,
    nD: int,
    tile_vals,
    out_s8,
    stream=None,
    threads: int = 256,
):
    """Scatter spherical ERI tiles into packed s8 layout on the GPU."""

    import cupy as cp

    _require_cuda_ext()
    nao_sph = int(nao_sph)
    nA = int(nA)
    nB = int(nB)
    nC = int(nC)
    nD = int(nD)

    with stream_ctx(stream):
        task_ab = cp.asarray(tasks.task_spAB, dtype=cp.int32)
        task_cd = cp.asarray(tasks.task_spCD, dtype=cp.int32)
        task_ab = cp.ascontiguousarray(task_ab)
        task_cd = cp.ascontiguousarray(task_cd)

        shell_ao_start_sph = cp.asarray(shell_ao_start_sph, dtype=cp.int32)
        shell_ao_start_sph = cp.ascontiguousarray(shell_ao_start_sph)

        tile_vals = cp.asarray(tile_vals, dtype=cp.float64)
        tile_vals = cp.ascontiguousarray(tile_vals).ravel()

        out_s8 = cp.asarray(out_s8, dtype=cp.float64)
        out_s8 = cp.ascontiguousarray(out_s8).ravel()

        _ext.scatter_eri_tiles_sph_s8_inplace_device(
            task_ab,
            task_cd,
            shell_pairs.sp_A,
            shell_pairs.sp_B,
            shell_ao_start_sph,
            int(nao_sph),
            int(nA),
            int(nB),
            int(nC),
            int(nD),
            tile_vals,
            out_s8,
            int(threads),
            int(_stream_ptr(stream)),
            False,
        )
        return out_s8


def scatter_eri_tiles_sph_s4_inplace_device(
    tasks: TaskList,
    shell_pairs: DeviceShellPairs,
    *,
    shell_ao_start_sph,
    nao_sph: int,
    nA: int,
    nB: int,
    nC: int,
    nD: int,
    tile_vals,
    out_s4,
    stream=None,
    threads: int = 256,
):
    """Scatter spherical ERI tiles into packed s4 layout on the GPU."""

    import cupy as cp

    _require_cuda_ext()
    nao_sph = int(nao_sph)
    nA = int(nA)
    nB = int(nB)
    nC = int(nC)
    nD = int(nD)

    with stream_ctx(stream):
        task_ab = cp.asarray(tasks.task_spAB, dtype=cp.int32)
        task_cd = cp.asarray(tasks.task_spCD, dtype=cp.int32)
        task_ab = cp.ascontiguousarray(task_ab)
        task_cd = cp.ascontiguousarray(task_cd)

        shell_ao_start_sph = cp.asarray(shell_ao_start_sph, dtype=cp.int32)
        shell_ao_start_sph = cp.ascontiguousarray(shell_ao_start_sph)

        tile_vals = cp.asarray(tile_vals, dtype=cp.float64)
        tile_vals = cp.ascontiguousarray(tile_vals).ravel()

        out_s4 = cp.asarray(out_s4, dtype=cp.float64)
        out_s4 = cp.ascontiguousarray(out_s4).ravel()

        _ext.scatter_eri_tiles_sph_s4_inplace_device(
            task_ab,
            task_cd,
            shell_pairs.sp_A,
            shell_pairs.sp_B,
            shell_ao_start_sph,
            int(nao_sph),
            int(nA),
            int(nB),
            int(nC),
            int(nD),
            tile_vals,
            out_s4,
            int(threads),
            int(_stream_ptr(stream)),
            False,
        )
        return out_s4


def to_device_basis_ss(basis: BasisSoA):
    import cupy as cp

    cx = cp.asarray(basis.shell_cxyz[:, 0], dtype=cp.float64)
    cy = cp.asarray(basis.shell_cxyz[:, 1], dtype=cp.float64)
    cz = cp.asarray(basis.shell_cxyz[:, 2], dtype=cp.float64)
    shell_prim_start = cp.asarray(basis.shell_prim_start, dtype=cp.int32)
    shell_nprim = cp.asarray(basis.shell_nprim, dtype=cp.int32)
    prim_exp = cp.asarray(basis.prim_exp, dtype=cp.float64)
    prim_coef = cp.asarray(basis.prim_coef, dtype=cp.float64)

    return DeviceBasisSS(
        shell_cx=cp.ascontiguousarray(cx),
        shell_cy=cp.ascontiguousarray(cy),
        shell_cz=cp.ascontiguousarray(cz),
        shell_prim_start=cp.ascontiguousarray(shell_prim_start),
        shell_nprim=cp.ascontiguousarray(shell_nprim),
        prim_exp=cp.ascontiguousarray(prim_exp),
        prim_coef=cp.ascontiguousarray(prim_coef),
    )


def to_device_shell_pairs(shell_pairs: ShellPairs):
    import cupy as cp

    sp_A = cp.asarray(shell_pairs.sp_A, dtype=cp.int32)
    sp_B = cp.asarray(shell_pairs.sp_B, dtype=cp.int32)
    sp_npair = cp.asarray(shell_pairs.sp_npair, dtype=cp.int32)
    sp_pair_start = cp.asarray(shell_pairs.sp_pair_start, dtype=cp.int32)
    return DeviceShellPairs(
        sp_A=cp.ascontiguousarray(sp_A),
        sp_B=cp.ascontiguousarray(sp_B),
        sp_npair=cp.ascontiguousarray(sp_npair),
        sp_pair_start=cp.ascontiguousarray(sp_pair_start),
    )


def build_pair_tables_ss_device(basis: DeviceBasisSS, shell_pairs: DeviceShellPairs, *, stream=None, threads: int = 256):
    import cupy as cp

    _require_cuda_ext()

    with stream_ctx(stream):
        total_pair_prims = int(shell_pairs.sp_pair_start[-1].item())
        pair_eta = cp.empty((total_pair_prims,), dtype=cp.float64)
        pair_Px = cp.empty((total_pair_prims,), dtype=cp.float64)
        pair_Py = cp.empty((total_pair_prims,), dtype=cp.float64)
        pair_Pz = cp.empty((total_pair_prims,), dtype=cp.float64)
        pair_cK = cp.empty((total_pair_prims,), dtype=cp.float64)

        _ext.build_pair_tables_ss_inplace_device(
            basis.shell_cx,
            basis.shell_cy,
            basis.shell_cz,
            basis.shell_prim_start,
            basis.shell_nprim,
            basis.prim_exp,
            basis.prim_coef,
            shell_pairs.sp_A,
            shell_pairs.sp_B,
            shell_pairs.sp_pair_start,
            shell_pairs.sp_npair,
            pair_eta,
            pair_Px,
            pair_Py,
            pair_Pz,
            pair_cK,
            int(threads),
            int(_stream_ptr(stream)),
            False,
        )
    return DevicePairTables(pair_eta=pair_eta, pair_Px=pair_Px, pair_Py=pair_Py, pair_Pz=pair_Pz, pair_cK=pair_cK)


def schwarz_ssss_device(
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    fast_boys: bool = False,
):
    import cupy as cp

    _require_cuda_ext()
    with stream_ctx(stream):
        nsp = int(shell_pairs.sp_A.shape[0])
        sp_Q = cp.empty((nsp,), dtype=cp.float64)
        _ext.schwarz_ssss_inplace_device(
            shell_pairs.sp_pair_start,
            shell_pairs.sp_npair,
            pair_tables.pair_eta,
            pair_tables.pair_Px,
            pair_tables.pair_Py,
            pair_tables.pair_Pz,
            pair_tables.pair_cK,
            sp_Q,
            int(threads),
            int(_stream_ptr(stream)),
            False,
            bool(fast_boys),
        )
    return sp_Q


def eri_ssss_device(
    tasks: TaskList,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
    boys: str = "ref",
):
    import cupy as cp

    _require_cuda_ext()
    with stream_ctx(stream):
        task_ab = cp.asarray(tasks.task_spAB, dtype=cp.int32)
        task_cd = cp.asarray(tasks.task_spCD, dtype=cp.int32)
        task_ab = cp.ascontiguousarray(task_ab)
        task_cd = cp.ascontiguousarray(task_cd)

        eri_out = cp.empty((tasks.ntasks,), dtype=cp.float64)
        if tasks.ntasks == 0:
            return eri_out

        mode = mode.lower().strip()
        if mode not in ("block", "warp", "multiblock", "auto"):
            raise ValueError("mode must be one of: 'block', 'warp', 'multiblock', 'auto'")

        boys = boys.lower().strip()
        if boys not in ("ref", "fast"):
            raise ValueError("boys must be one of: 'ref', 'fast'")
        fast_boys = boys == "fast"

        if mode == "block":
            _ext.eri_ssss_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                eri_out,
                int(threads),
                int(_stream_ptr(stream)),
                False,
                bool(fast_boys),
            )
            return eri_out

        if mode == "warp":
            _ext.eri_ssss_warp_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                eri_out,
                int(threads),
                int(_stream_ptr(stream)),
                False,
                bool(fast_boys),
            )
            return eri_out

        if mode == "multiblock":
            if blocks_per_task <= 0:
                raise ValueError("blocks_per_task must be > 0")
            partial = cp.empty((tasks.ntasks * int(blocks_per_task),), dtype=cp.float64)
            _ext.eri_ssss_multiblock_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                partial,
                int(blocks_per_task),
                eri_out,
                int(threads),
                int(_stream_ptr(stream)),
                False,
                bool(fast_boys),
            )
            return eri_out

        # mode == "auto": bin by a crude work proxy.
        if work_small_max < 0 or work_large_min < 0:
            raise ValueError("work_small_max/work_large_min must be >= 0")
        if work_small_max >= work_large_min:
            raise ValueError("work_small_max must be < work_large_min")

        work = shell_pairs.sp_npair[task_ab].astype(cp.int64) * shell_pairs.sp_npair[task_cd].astype(cp.int64)
        small_mask = work <= int(work_small_max)
        large_mask = work >= int(work_large_min)
        med_mask = ~(small_mask | large_mask)

        small_idx = cp.nonzero(small_mask)[0]
        med_idx = cp.nonzero(med_mask)[0]
        large_idx = cp.nonzero(large_mask)[0]

        if int(small_idx.size) > 0:
            ab_small = cp.ascontiguousarray(task_ab[small_idx])
            cd_small = cp.ascontiguousarray(task_cd[small_idx])
            out_small = cp.empty((int(ab_small.shape[0]),), dtype=cp.float64)
            _ext.eri_ssss_warp_inplace_device(
                ab_small,
                cd_small,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out_small,
                int(threads),
                int(_stream_ptr(stream)),
                False,
                bool(fast_boys),
            )
            eri_out[small_idx] = out_small

        if int(med_idx.size) > 0:
            ab_med = cp.ascontiguousarray(task_ab[med_idx])
            cd_med = cp.ascontiguousarray(task_cd[med_idx])
            out_med = cp.empty((int(ab_med.shape[0]),), dtype=cp.float64)
            _ext.eri_ssss_inplace_device(
                ab_med,
                cd_med,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out_med,
                int(threads),
                int(_stream_ptr(stream)),
                False,
                bool(fast_boys),
            )
            eri_out[med_idx] = out_med

        if int(large_idx.size) > 0:
            if blocks_per_task <= 0:
                raise ValueError("blocks_per_task must be > 0")
            ab_large = cp.ascontiguousarray(task_ab[large_idx])
            cd_large = cp.ascontiguousarray(task_cd[large_idx])
            out_large = cp.empty((int(ab_large.shape[0]),), dtype=cp.float64)
            partial = cp.empty((int(ab_large.shape[0]) * int(blocks_per_task),), dtype=cp.float64)
            _ext.eri_ssss_multiblock_inplace_device(
                ab_large,
                cd_large,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                partial,
                int(blocks_per_task),
                out_large,
                int(threads),
                int(_stream_ptr(stream)),
                False,
                bool(fast_boys),
            )
            eri_out[large_idx] = out_large
        return eri_out


def df_metric_2c2e_ss_device(
    aux_basis: BasisSoA,
    *,
    stream=None,
    threads: int = 256,
    boys: str = "ref",
):
    """Compute DF metric V(P,Q) = (P|Q) for s-shell auxiliary bases on GPU.

    This implementation reuses the Step-1 (ss|ss) evaluator by introducing a dummy
    constant function 1 on each electron:
      (P|Q) == (P*1 | Q*1)
    """

    import cupy as cp

    _require_cuda_ext()
    with stream_ctx(stream):
        aux_ext, dummy_shell = _basis_with_dummy_constant_shell(aux_basis)
        sp_aux = _shell_pairs_dummy_times_real(aux_ext, dummy_shell)
        naux = int(sp_aux.sp_A.shape[0])
        if naux == 0:
            return cp.empty((0, 0), dtype=cp.float64)

        dbasis = to_device_basis_ss(aux_ext)
        dsp = to_device_shell_pairs(sp_aux)
        pt = build_pair_tables_ss_device(dbasis, dsp, stream=stream, threads=threads)

        task_ab = np.repeat(np.arange(naux, dtype=np.int32), naux)
        task_cd = np.tile(np.arange(naux, dtype=np.int32), naux)
        tasks = TaskList(task_spAB=task_ab, task_spCD=task_cd)
        eri = eri_ssss_device(tasks, dsp, pt, stream=stream, threads=threads, mode="auto", boys=boys)
        V = eri.reshape((naux, naux))
        return 0.5 * (V + V.T)


def df_int3c2e_ss_device(
    ao_basis: BasisSoA,
    aux_basis: BasisSoA,
    *,
    stream=None,
    threads: int = 256,
    boys: str = "ref",
):
    """Compute 3c2e integrals X(μ,ν,P) = (μν|P) for s-shell AO/aux bases on GPU.

    Implementation uses the Step-1 (ss|ss) evaluator with a dummy constant function:
      (μν|P) == (μν | P*1)
    """

    import cupy as cp

    _require_cuda_ext()
    with stream_ctx(stream):
        nao = int(ao_basis.shell_cxyz.shape[0])
        if nao == 0:
            return cp.empty((0, 0, 0), dtype=cp.float64)

        sp_ao = build_shell_pairs(ao_basis)
        nsp_ao = int(sp_ao.sp_A.shape[0])

        aux_ext, dummy_shell = _basis_with_dummy_constant_shell(aux_basis)
        sp_aux = _shell_pairs_dummy_times_real(aux_ext, dummy_shell)
        naux = int(sp_aux.sp_A.shape[0])

        if naux == 0:
            return cp.empty((nao, nao, 0), dtype=cp.float64)

        # Pair tables for AO pairs
        dbasis_ao = to_device_basis_ss(ao_basis)
        dsp_ao = to_device_shell_pairs(sp_ao)
        pt_ao = build_pair_tables_ss_device(dbasis_ao, dsp_ao, stream=stream, threads=threads)

        # Pair tables for aux "P*1" pairs
        dbasis_aux = to_device_basis_ss(aux_ext)
        dsp_aux = to_device_shell_pairs(sp_aux)
        pt_aux = build_pair_tables_ss_device(dbasis_aux, dsp_aux, stream=stream, threads=threads)

        # Combine (AO pairs) + (aux pairs) into one pair-table list for the ERI kernel.
        total_pair_ao = int(dsp_ao.sp_pair_start[-1].item())
        offset = cp.int32(total_pair_ao)

        sp_npair = cp.concatenate([dsp_ao.sp_npair, dsp_aux.sp_npair]).astype(cp.int32)
        sp_pair_start = cp.concatenate(
            [dsp_ao.sp_pair_start[:-1], (dsp_aux.sp_pair_start + offset).astype(cp.int32)]
        ).astype(cp.int32)

        pair_eta = cp.concatenate([pt_ao.pair_eta, pt_aux.pair_eta])
        pair_Px = cp.concatenate([pt_ao.pair_Px, pt_aux.pair_Px])
        pair_Py = cp.concatenate([pt_ao.pair_Py, pt_aux.pair_Py])
        pair_Pz = cp.concatenate([pt_ao.pair_Pz, pt_aux.pair_Pz])
        pair_cK = cp.concatenate([pt_ao.pair_cK, pt_aux.pair_cK])

        nsp_total = int(nsp_ao + naux)
        shell_pairs = DeviceShellPairs(
            sp_A=cp.empty((nsp_total,), dtype=cp.int32),
            sp_B=cp.empty((nsp_total,), dtype=cp.int32),
            sp_npair=cp.ascontiguousarray(sp_npair),
            sp_pair_start=cp.ascontiguousarray(sp_pair_start),
        )
        pair_tables = DevicePairTables(
            pair_eta=cp.ascontiguousarray(pair_eta),
            pair_Px=cp.ascontiguousarray(pair_Px),
            pair_Py=cp.ascontiguousarray(pair_Py),
            pair_Pz=cp.ascontiguousarray(pair_Pz),
            pair_cK=cp.ascontiguousarray(pair_cK),
        )

        # Tasks: all (AO shell pair, aux shell) combinations.
        task_ab = np.repeat(np.arange(nsp_ao, dtype=np.int32), naux)
        task_cd = (nsp_ao + np.tile(np.arange(naux, dtype=np.int32), nsp_ao)).astype(np.int32, copy=False)
        tasks = TaskList(task_spAB=task_ab, task_spCD=task_cd)
        eri = eri_ssss_device(tasks, shell_pairs, pair_tables, stream=stream, threads=threads, mode="auto", boys=boys)
        eri = eri.reshape((nsp_ao, naux))

        # Expand canonical shell pairs (A>=B) to full (μ,ν) with symmetry.
        spA = cp.asarray(sp_ao.sp_A, dtype=cp.int64)
        spB = cp.asarray(sp_ao.sp_B, dtype=cp.int64)
        idx_ab = spA * int(nao) + spB
        idx_ba = spB * int(nao) + spA

        X_flat = cp.zeros((nao * nao, naux), dtype=cp.float64)
        X_flat[idx_ab, :] = eri
        X_flat[idx_ba, :] = eri
        return X_flat.reshape((nao, nao, naux))


def df_metric_2c2e_sp_device(
    aux_basis,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "auto",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 4,
):
    """Compute DF metric V(P,Q) = (P|Q) on GPU for s/p-only auxiliary bases (cart).

    This uses Step-2 4c2e class kernels by representing the 2c2e metric as:
      (P|Q) == (P*1 | Q*1)
    where "1" is modeled as a dummy s-shell with one primitive exp=0, coef=1.
    """

    import cupy as cp

    _require_cuda_ext()

    shell_l = np.asarray(aux_basis.shell_l, dtype=np.int32).ravel()
    if shell_l.size == 0:
        return cp.empty((0, 0), dtype=cp.float64)
    if int(shell_l.max()) > 1:
        raise NotImplementedError("df_metric_2c2e_sp_device currently supports only l<=1 (s/p) aux shells")
    return df_metric_2c2e_rys_device(
        aux_basis,
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def _get_df_metric_2c2e_rys_plan(
    aux_basis,
    *,
    threads: int,
    mode: str,
    work_small_max: int,
    work_large_min: int,
    blocks_per_task: int,
) -> _DFMetric2c2eRysPlan:
    import cupy as cp

    _require_cuda_ext()

    dev = int(cp.cuda.runtime.getDevice())
    key = (dev, id(aux_basis), int(threads), str(mode), int(work_small_max), int(work_large_min), int(blocks_per_task))
    hit = _df_metric_2c2e_rys_plan_cache.get(key)
    if hit is not None:
        aux_ref, plan = hit
        if aux_ref() is aux_basis:
            return plan
        del _df_metric_2c2e_rys_plan_cache[key]

    shell_l = np.asarray(aux_basis.shell_l, dtype=np.int32).ravel()
    if shell_l.size == 0:
        raise ValueError("aux_basis must be non-empty")
    if int(shell_l.max()) > CUDA_MAX_L:
        raise NotImplementedError(
            f"_get_df_metric_2c2e_rys_plan currently supports only l<={CUDA_MAX_L} per aux shell"
        )

    # Build combined aux+dummy basis arrays.
    n_shell = int(aux_basis.shell_cxyz.shape[0])
    n_prim = int(aux_basis.prim_exp.shape[0])
    dummy_shell = n_shell

    shell_cxyz = np.concatenate([np.asarray(aux_basis.shell_cxyz, dtype=np.float64), np.zeros((1, 3), dtype=np.float64)])
    shell_prim_start = np.concatenate([np.asarray(aux_basis.shell_prim_start, dtype=np.int32), np.asarray([n_prim], dtype=np.int32)])
    shell_nprim = np.concatenate([np.asarray(aux_basis.shell_nprim, dtype=np.int32), np.asarray([1], dtype=np.int32)])
    prim_exp = np.concatenate([np.asarray(aux_basis.prim_exp, dtype=np.float64), np.asarray([0.0], dtype=np.float64)])
    prim_coef = np.concatenate([np.asarray(aux_basis.prim_coef, dtype=np.float64), np.asarray([1.0], dtype=np.float64)])
    shell_l_comb = np.concatenate([shell_l, np.asarray([0], dtype=np.int32)])

    basis_like = SimpleNamespace(
        shell_cxyz=shell_cxyz,
        shell_prim_start=shell_prim_start,
        shell_nprim=shell_nprim,
        prim_exp=prim_exp,
        prim_coef=prim_coef,
    )

    # ShellPairs: (P, dummy) for each aux shell P.
    sp_A = np.arange(n_shell, dtype=np.int32)
    sp_B = np.full((n_shell,), int(dummy_shell), dtype=np.int32)
    sp_npair = np.asarray(aux_basis.shell_nprim, dtype=np.int32)
    sp_pair_start = np.empty((n_shell + 1,), dtype=np.int32)
    sp_pair_start[0] = 0
    sp_pair_start[1:] = np.cumsum(sp_npair, dtype=np.int32)
    sp = ShellPairs(sp_A=sp_A, sp_B=sp_B, sp_npair=sp_npair, sp_pair_start=sp_pair_start)

    dbasis = to_device_basis_ss(basis_like)
    dsp = to_device_shell_pairs(sp)
    pt = build_pair_tables_ss_device(dbasis, dsp, threads=threads)

    # Unique unordered (P,Q) pairs; orient each task so that l(P) >= l(Q) to maximize kernel coverage.
    task_ab: list[int] = []
    task_cd: list[int] = []
    for i in range(n_shell):
        li = int(shell_l[i])
        for j in range(i + 1):
            lj = int(shell_l[j])
            hi, lo = (i, j) if li >= lj else (j, i)
            task_ab.append(int(hi))
            task_cd.append(int(lo))

    tasks = TaskList(task_spAB=np.asarray(task_ab, dtype=np.int32), task_spCD=np.asarray(task_cd, dtype=np.int32))
    tasks = with_task_class_id(tasks, sp, shell_l_comb)

    ao_start = np.asarray(aux_basis.shell_ao_start, dtype=np.int32)
    nfunc = np.asarray([ncart(int(l)) for l in shell_l], dtype=np.int32)
    naux = int(np.max(ao_start + nfunc))

    from .eri_dispatch import plan_kernel_batches_spd

    batches = plan_kernel_batches_spd(tasks, shell_pairs=sp, shell_l=shell_l_comb)

    batch_p0_dev: list[object] = []
    batch_q0_dev: list[object] = []
    for batch in batches:
        idx = np.asarray(batch.task_idx, dtype=np.int32).ravel()
        spab = np.asarray(tasks.task_spAB[idx], dtype=np.int32).ravel()
        spcd = np.asarray(tasks.task_spCD[idx], dtype=np.int32).ravel()
        P_shell = np.asarray(sp.sp_A[spab], dtype=np.int32).ravel()
        Q_shell = np.asarray(sp.sp_A[spcd], dtype=np.int32).ravel()
        p0 = np.asarray(aux_basis.shell_ao_start[P_shell], dtype=np.int32).ravel()
        q0 = np.asarray(aux_basis.shell_ao_start[Q_shell], dtype=np.int32).ravel()
        batch_p0_dev.append(cp.ascontiguousarray(cp.asarray(p0, dtype=cp.int32)))
        batch_q0_dev.append(cp.ascontiguousarray(cp.asarray(q0, dtype=cp.int32)))

    cp.cuda.get_current_stream().synchronize()

    plan = _DFMetric2c2eRysPlan(
        dbasis=dbasis,
        dsp=dsp,
        pt=pt,
        batches=batches,
        batch_p0_dev=batch_p0_dev,
        batch_q0_dev=batch_q0_dev,
        naux=int(naux),
    )
    _df_metric_2c2e_rys_plan_cache[key] = (weakref.ref(aux_basis), plan)
    _trim_plan_cache(_df_metric_2c2e_rys_plan_cache)
    return plan


def _get_df_int3c2e_rys_plan(
    ao_basis,
    aux_basis,
    *,
    threads: int,
    mode: str,
    work_small_max: int,
    work_large_min: int,
    blocks_per_task: int,
) -> _DFInt3c2eRysPlan:
    import cupy as cp

    _require_cuda_ext()

    dev = int(cp.cuda.runtime.getDevice())
    key = (
        dev,
        id(ao_basis),
        id(aux_basis),
        int(threads),
        str(mode),
        int(work_small_max),
        int(work_large_min),
        int(blocks_per_task),
    )
    hit = _df_int3c2e_rys_plan_cache.get(key)
    if hit is not None:
        ao_ref, aux_ref, plan = hit
        if ao_ref() is ao_basis and aux_ref() is aux_basis:
            return plan
        del _df_int3c2e_rys_plan_cache[key]

    ao_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    aux_l = np.asarray(aux_basis.shell_l, dtype=np.int32).ravel()
    if ao_l.size == 0 or aux_l.size == 0:
        raise ValueError("ao_basis and aux_basis must be non-empty")
    if int(ao_l.max()) > CUDA_MAX_L or int(aux_l.max()) > CUDA_MAX_L:
        raise NotImplementedError(
            f"_get_df_int3c2e_rys_plan currently supports only l<={CUDA_MAX_L} for AO and aux shells"
        )

    # Combined (AO + aux + dummy) basis arrays.
    n_shell_ao = int(ao_basis.shell_cxyz.shape[0])
    n_shell_aux = int(aux_basis.shell_cxyz.shape[0])
    dummy_shell = n_shell_ao + n_shell_aux

    ao_prim_n = int(ao_basis.prim_exp.shape[0])
    aux_prim_n = int(aux_basis.prim_exp.shape[0])

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

    dbasis = to_device_basis_ss(basis_like)
    dsp = to_device_shell_pairs(sp_all)
    pt = build_pair_tables_ss_device(dbasis, dsp, threads=threads)

    # Tasks: all (AO shell pair, aux shell) combinations for full aux coverage.
    task_ab = np.tile(np.arange(nsp_ao, dtype=np.int32), int(n_shell_aux))
    task_cd = np.repeat((np.int32(nsp_ao) + np.arange(n_shell_aux, dtype=np.int32)), int(nsp_ao))
    tasks = TaskList(task_spAB=task_ab, task_spCD=task_cd)
    tasks = with_task_class_id(tasks, sp_all, shell_l)

    from .eri_dispatch import plan_kernel_batches_spd

    batches = plan_kernel_batches_spd(tasks, shell_pairs=sp_all, shell_l=shell_l)

    ao_start = np.asarray(ao_basis.shell_ao_start, dtype=np.int32)
    ao_nfunc = np.asarray([ncart(int(l)) for l in ao_l], dtype=np.int32)
    nao = int(np.max(ao_start + ao_nfunc))

    # Spherical AO layout for the AO basis shells (real spherical harmonics).
    # For expanded cart bases (BasisCartSoA), each logical shell has one contraction (nctr=1).
    shell_ao_start_sph = np.empty((int(n_shell_ao),), dtype=np.int32)
    cursor = 0
    for sh in range(int(n_shell_ao)):
        shell_ao_start_sph[sh] = np.int32(cursor)
        cursor += int(int(ao_l[sh]) * 2 + 1)
    nao_sph = int(cursor)

    aux_start = np.asarray(aux_basis.shell_ao_start, dtype=np.int32)
    aux_nfunc = np.asarray([ncart(int(l)) for l in aux_l], dtype=np.int32)
    naux = int(np.max(aux_start + aux_nfunc))

    batch_a0_dev: list[object] = []
    batch_b0_dev: list[object] = []
    batch_a0_sph_dev: list[object] = []
    batch_b0_sph_dev: list[object] = []
    batch_p0_dev: list[object] = []
    batch_nB: list[int] = []
    batch_la: list[int] = []
    batch_lb: list[int] = []

    for batch in batches:
        idx = np.asarray(batch.task_idx, dtype=np.int32).ravel()
        if idx.size == 0:
            batch_a0_dev.append(cp.empty((0,), dtype=cp.int32))
            batch_b0_dev.append(cp.empty((0,), dtype=cp.int32))
            batch_a0_sph_dev.append(cp.empty((0,), dtype=cp.int32))
            batch_b0_sph_dev.append(cp.empty((0,), dtype=cp.int32))
            batch_p0_dev.append(cp.empty((0,), dtype=cp.int32))
            batch_nB.append(0)
            batch_la.append(0)
            batch_lb.append(0)
            continue

        spab = np.asarray(tasks.task_spAB[idx], dtype=np.int32).ravel()
        spcd = np.asarray(tasks.task_spCD[idx], dtype=np.int32).ravel()

        A = np.asarray(sp_all.sp_A[spab], dtype=np.int32).ravel()
        B = np.asarray(sp_all.sp_B[spab], dtype=np.int32).ravel()
        P = (np.asarray(sp_all.sp_A[spcd], dtype=np.int32).ravel() - np.int32(n_shell_ao)).astype(np.int32, copy=False)

        a0 = np.asarray(ao_basis.shell_ao_start[A], dtype=np.int32).ravel()
        b0 = np.asarray(ao_basis.shell_ao_start[B], dtype=np.int32).ravel()
        p0 = np.asarray(aux_basis.shell_ao_start[P], dtype=np.int32).ravel()

        a0_sph = np.asarray(shell_ao_start_sph[A], dtype=np.int32).ravel()
        b0_sph = np.asarray(shell_ao_start_sph[B], dtype=np.int32).ravel()

        batch_a0_dev.append(cp.ascontiguousarray(cp.asarray(a0, dtype=cp.int32)))
        batch_b0_dev.append(cp.ascontiguousarray(cp.asarray(b0, dtype=cp.int32)))
        batch_a0_sph_dev.append(cp.ascontiguousarray(cp.asarray(a0_sph, dtype=cp.int32)))
        batch_b0_sph_dev.append(cp.ascontiguousarray(cp.asarray(b0_sph, dtype=cp.int32)))
        batch_p0_dev.append(cp.ascontiguousarray(cp.asarray(p0, dtype=cp.int32)))

        la = int(ao_l[int(A[0])])
        lb = int(ao_l[int(B[0])])
        batch_nB.append(int(ncart(lb)))
        batch_la.append(int(la))
        batch_lb.append(int(lb))

    cp.cuda.get_current_stream().synchronize()

    plan = _DFInt3c2eRysPlan(
        dbasis=dbasis,
        dsp=dsp,
        pt=pt,
        batches=batches,
        batch_a0_dev=batch_a0_dev,
        batch_b0_dev=batch_b0_dev,
        batch_a0_sph_dev=batch_a0_sph_dev,
        batch_b0_sph_dev=batch_b0_sph_dev,
        batch_p0_dev=batch_p0_dev,
        batch_nB=batch_nB,
        batch_la=batch_la,
        batch_lb=batch_lb,
        nao=int(nao),
        nao_sph=int(nao_sph),
        naux=int(naux),
    )
    _df_int3c2e_rys_plan_cache[key] = (weakref.ref(ao_basis), weakref.ref(aux_basis), plan)
    _trim_plan_cache(_df_int3c2e_rys_plan_cache)
    return plan


def _get_df_int3c2e_rys_contracted_ao_plan(
    ao_basis,
    aux_basis,
    *,
    threads: int,
) -> _DFInt3c2eRysContractedAOPlan:
    """Build a cached DF int3c2e plan for *contracted* AO bases (non-expanded nctr)."""

    import cupy as cp

    _require_cuda_ext()

    dev = int(cp.cuda.runtime.getDevice())
    key = (dev, id(ao_basis), id(aux_basis), int(threads))
    hit = _df_int3c2e_rys_contracted_ao_plan_cache.get(key)
    if hit is not None:
        ao_ref, aux_ref, plan = hit
        if ao_ref() is ao_basis and aux_ref() is aux_basis:
            return plan
        del _df_int3c2e_rys_contracted_ao_plan_cache[key]

    if not hasattr(ao_basis, "shell_nctr") or not hasattr(ao_basis, "shell_coef_start") or not hasattr(ao_basis, "prim_coef_flat"):
        raise TypeError("ao_basis must be a contracted basis (expected shell_nctr/shell_coef_start/prim_coef_flat)")
    if hasattr(aux_basis, "shell_nctr") or hasattr(aux_basis, "prim_coef_flat"):
        raise NotImplementedError("contracted aux bases are not supported yet; provide an expanded aux basis (BasisCartSoA)")

    ao_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    aux_l = np.asarray(aux_basis.shell_l, dtype=np.int32).ravel()
    if ao_l.size == 0 or aux_l.size == 0:
        raise ValueError("ao_basis and aux_basis must be non-empty")
    if int(ao_l.max()) > CUDA_MAX_L or int(aux_l.max()) > CUDA_MAX_L:
        raise NotImplementedError(
            f"_get_df_int3c2e_rys_contracted_ao_plan supports only l<={CUDA_MAX_L} for AO and aux shells"
        )

    ao_nctr = np.asarray(ao_basis.shell_nctr, dtype=np.int32).ravel()
    if ao_nctr.size != ao_l.size:
        raise ValueError("ao_basis.shell_nctr shape mismatch")
    if int(ao_nctr.max()) > CUDA_MAX_L:
        raise NotImplementedError(
            f"contracted AO DF path currently supports nctr<={CUDA_MAX_L} per shell; expand contractions for larger nctr"
        )

    # Combined (AO + aux + dummy) basis arrays.
    n_shell_ao = int(ao_basis.shell_cxyz.shape[0])
    n_shell_aux = int(aux_basis.shell_cxyz.shape[0])
    dummy_shell = n_shell_ao + n_shell_aux

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
    # Pair tables should contain only Kab geometry on the AO side; contraction coefficients are applied inside the DF kernel.
    prim_coef = np.concatenate(
        [
            np.ones((ao_prim_n,), dtype=np.float64),
            np.asarray(aux_basis.prim_coef, dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
        ]
    )
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

    dbasis = to_device_basis_ss(basis_like)
    dsp = to_device_shell_pairs(sp_all)
    pt = build_pair_tables_ss_device(dbasis, dsp, threads=int(threads))

    # Tasks: all (AO shell pair, aux shell) combinations for full aux coverage.
    task_ab = np.tile(np.arange(nsp_ao, dtype=np.int32), int(n_shell_aux))
    task_cd = np.repeat((np.int32(nsp_ao) + np.arange(n_shell_aux, dtype=np.int32)), int(nsp_ao))
    tasks = TaskList(task_spAB=task_ab, task_spCD=task_cd)
    tasks = with_task_class_id(tasks, sp_all, shell_l)

    perm, class_ids, offsets = group_tasks_by_class(tasks.task_class_id)

    ao_start = np.asarray(ao_basis.shell_ao_start, dtype=np.int32)
    ao_nfunc = (np.asarray([ncart(int(l)) for l in ao_l], dtype=np.int32) * ao_nctr).astype(np.int32, copy=False)
    nao = int(np.max(ao_start + ao_nfunc))

    aux_start = np.asarray(aux_basis.shell_ao_start, dtype=np.int32)
    aux_nfunc = np.asarray([ncart(int(l)) for l in aux_l], dtype=np.int32)
    naux = int(np.max(aux_start + aux_nfunc))

    ao_shell_ao_start_dev = cp.ascontiguousarray(cp.asarray(ao_start, dtype=cp.int32))
    ao_shell_nctr_dev = cp.ascontiguousarray(cp.asarray(ao_nctr, dtype=cp.int32))
    ao_shell_coef_start_dev = cp.ascontiguousarray(cp.asarray(np.asarray(ao_basis.shell_coef_start, dtype=np.int32), dtype=cp.int32))
    ao_prim_coef_dev = cp.ascontiguousarray(cp.asarray(np.asarray(ao_basis.prim_coef_flat, dtype=np.float64), dtype=cp.float64))
    aux_shell_ao_start_dev = cp.ascontiguousarray(cp.asarray(aux_start, dtype=cp.int32))

    cp.cuda.get_current_stream().synchronize()

    plan = _DFInt3c2eRysContractedAOPlan(
        dbasis=dbasis,
        dsp=dsp,
        pt=pt,
        tasks=tasks,
        perm=perm,
        class_ids=class_ids,
        offsets=offsets,
        ao_shell_ao_start_dev=ao_shell_ao_start_dev,
        ao_shell_nctr_dev=ao_shell_nctr_dev,
        ao_shell_coef_start_dev=ao_shell_coef_start_dev,
        ao_prim_coef_dev=ao_prim_coef_dev,
        aux_shell_ao_start_dev=aux_shell_ao_start_dev,
        n_shell_ao=int(n_shell_ao),
        nao=int(nao),
        naux=int(naux),
    )
    _df_int3c2e_rys_contracted_ao_plan_cache[key] = (weakref.ref(ao_basis), weakref.ref(aux_basis), plan)
    _trim_plan_cache(_df_int3c2e_rys_contracted_ao_plan_cache)
    return plan


def df_metric_2c2e_rys_device(
    aux_basis,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "auto",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 4,
):
    """Compute DF metric V(P,Q) = (P|Q) on GPU for Cartesian auxiliary bases (general-l, reference-oriented).

    Uses the Step-2 4c2e evaluator via the identity:
      (P|Q) == (P*1 | Q*1)
    where "1" is modeled as a dummy s-shell with one primitive exp=0, coef=1.

    Current limitation
    - requires `aux_basis` to provide `shell_ao_start` (i.e., `BasisCartSoA`)
    - CUDA backend supports `l<=CUDA_MAX_L` per shell (nroots<=CUDA_MAX_NROOTS)
    """

    import cupy as cp

    _require_cuda_ext()
    if not hasattr(aux_basis, "shell_ao_start"):
        raise TypeError("aux_basis must provide shell_ao_start (use a packed cartesian basis like BasisCartSoA)")

    with stream_ctx(stream):
        shell_l = np.asarray(aux_basis.shell_l, dtype=np.int32).ravel()
        if shell_l.size == 0:
            return cp.empty((0, 0), dtype=cp.float64)
        if int(shell_l.max()) > CUDA_MAX_L:
            raise NotImplementedError(
                f"df_metric_2c2e_rys_device currently supports only l<={CUDA_MAX_L} per aux shell"
            )

        mode = mode.lower().strip()
        if mode not in ("block", "warp", "multiblock", "auto"):
            raise ValueError("mode must be one of: 'block', 'warp', 'multiblock', 'auto'")

        plan = _get_df_metric_2c2e_rys_plan(
            aux_basis,
            threads=int(threads),
            mode=str(mode),
            work_small_max=int(work_small_max),
            work_large_min=int(work_large_min),
            blocks_per_task=int(blocks_per_task),
        )

        naux = int(plan.naux)
        V = cp.zeros((naux, naux), dtype=cp.float64)

        from .eri_dispatch import run_kernel_batch_spd

        for batch, p0_dev, q0_dev in zip(plan.batches, plan.batch_p0_dev, plan.batch_q0_dev):
            tile = run_kernel_batch_spd(
                batch,
                dbasis=plan.dbasis,
                dsp=plan.dsp,
                pt=plan.pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )
            nP = int(tile.shape[1])
            nQ = int(tile.shape[2])
            _ext.scatter_df_metric_tiles_inplace_device(
                tile.ravel(),
                p0_dev,
                q0_dev,
                int(naux),
                int(nP),
                int(nQ),
                V.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )

        return V


def df_int3c2e_sp_device(
    ao_basis,
    aux_basis,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "auto",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 4,
    ao_contract_mode: str = "auto",
    ao_rep: str = "cart",
    profile: dict | None = None,
):
    """Compute 3c2e integrals X(μ,ν,P) = (μν|P) on GPU for s/p-only AO+aux bases (cart).

    Uses Step-2 4c2e class kernels by representing the 3c2e integral as:
      (μν|P) == (μν | P*1)
    where "1" is modeled as a dummy s-shell with one primitive exp=0, coef=1.
    """

    import cupy as cp

    _require_cuda_ext()

    ao_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    aux_l = np.asarray(aux_basis.shell_l, dtype=np.int32).ravel()
    if ao_l.size == 0 or aux_l.size == 0:
        nao = 0 if ao_l.size == 0 else int(np.max(np.asarray(ao_basis.shell_ao_start, dtype=np.int32)))
        return cp.empty((nao, nao, 0), dtype=cp.float64)
    if int(ao_l.max()) > 1 or int(aux_l.max()) > 1:
        raise NotImplementedError("df_int3c2e_sp_device currently supports only l<=1 (s/p) for AO and aux shells")
    return df_int3c2e_rys_device(
        ao_basis,
        aux_basis,
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
        ao_contract_mode=ao_contract_mode,
        ao_rep=ao_rep,
        profile=profile,
    )


def df_int3c2e_rys_device(
    ao_basis,
    aux_basis,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "auto",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 4,
    ao_contract_mode: str = "auto",
    ao_rep: str = "cart",
    profile: dict | None = None,
):
    """Compute 3c2e integrals X(μ,ν,P) = (μν|P) on GPU for Cartesian AO+aux bases (general-l, reference-oriented).

    This is the full-tensor convenience wrapper. For streaming consumers, prefer
    :func:`df_int3c2e_rys_device_block` to compute a shell-block of aux functions.
    """

    n_shell_aux = int(aux_basis.shell_cxyz.shape[0])
    return df_int3c2e_rys_device_block(
        ao_basis,
        aux_basis,
        aux_shell_start=0,
        aux_shell_stop=n_shell_aux,
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
        ao_contract_mode=ao_contract_mode,
        ao_rep=ao_rep,
        profile=profile,
    )


def df_int3c2e_rys_contracted_ao_device_block(
    ao_basis,
    aux_basis,
    *,
    aux_shell_start: int,
    aux_shell_stop: int,
    stream=None,
    threads: int = 256,
    profile: dict | None = None,
):
    """Compute DF 3c2e X(mu,nu,P) on GPU with AO general contractions (non-expanded nctr).

    Notes
    - AO basis must provide: shell_nctr, shell_coef_start, prim_coef_flat (prim-major coef matrix).
    - Aux basis is currently required to be expanded (BasisCartSoA-like, nctr==1 view).
    - This path writes directly into X and does not use the tile+scatter staging.
    """

    import cupy as cp

    _require_cuda_ext()
    with stream_ctx(stream):
        if not hasattr(ao_basis, "shell_nctr") or not hasattr(ao_basis, "shell_coef_start") or not hasattr(ao_basis, "prim_coef_flat"):
            raise TypeError("ao_basis must be a contracted AO basis (expected shell_nctr/shell_coef_start/prim_coef_flat)")
        if hasattr(aux_basis, "shell_nctr") or hasattr(aux_basis, "prim_coef_flat"):
            raise NotImplementedError("contracted aux bases are not supported yet; provide an expanded aux basis (BasisCartSoA)")

        ao_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
        aux_l = np.asarray(aux_basis.shell_l, dtype=np.int32).ravel()
        if ao_l.size == 0 or aux_l.size == 0:
            return cp.empty((0, 0, 0), dtype=cp.float64)
        if int(ao_l.max()) > CUDA_MAX_L or int(aux_l.max()) > CUDA_MAX_L:
            raise NotImplementedError(
                f"df_int3c2e_rys_contracted_ao_device_block currently supports only l<={CUDA_MAX_L} for AO and aux shells"
            )

        ao_nctr = np.asarray(ao_basis.shell_nctr, dtype=np.int32).ravel()
        if int(ao_nctr.max()) > CUDA_MAX_L:
            raise NotImplementedError(
                f"contracted AO DF path currently supports nctr<={CUDA_MAX_L} per shell; expand contractions for larger nctr"
            )

        aux_shell_start = int(aux_shell_start)
        aux_shell_stop = int(aux_shell_stop)
        if aux_shell_start < 0 or aux_shell_stop < aux_shell_start or aux_shell_stop > int(aux_l.size):
            raise ValueError("aux_shell_start/aux_shell_stop out of range")

        # AO size (contracted): nAO(shell) = ncart(l) * nctr.
        ao_start = np.asarray(ao_basis.shell_ao_start, dtype=np.int32)
        ao_nfunc = (np.asarray([ncart(int(l)) for l in ao_l], dtype=np.int32) * ao_nctr).astype(np.int32, copy=False)
        nao = int(np.max(ao_start + ao_nfunc))

        # Fast path: full aux coverage uses a cached plan (pair tables + task grouping + AO coef H2D).
        if aux_shell_start == 0 and aux_shell_stop == int(aux_l.size):
            s0 = cp.cuda.get_current_stream()
            if profile is not None:
                t0 = cp.cuda.Event()
                t1 = cp.cuda.Event()
                t0.record(s0)
            plan = _get_df_int3c2e_rys_contracted_ao_plan(ao_basis, aux_basis, threads=int(threads))
            if profile is not None:
                t1.record(s0)
                t1.synchronize()
                prof = profile.setdefault("df_int3c2e", {})
                prof["plan_ms"] = float(cp.cuda.get_elapsed_time(t0, t1))

            # Common contracted AO bases (e.g., cc-pVXZ) have nctr<=2; use the CTR_MAX=2
            # specialization to reduce register pressure in the DF kernel.
            use_ctr2 = int(ao_nctr.max()) <= 2

            X = cp.zeros((int(plan.nao), int(plan.nao), int(plan.naux)), dtype=cp.float64)
            dispatch_profile = None
            if profile is not None:
                dispatch_profile = profile.setdefault("df_int3c2e", {}).setdefault("kernel_dispatch", {})

            # Hybrid path: reuse existing Step-2 kernels for Step-2-eligible task classes by
            # temporarily building an expanded (nctr==1) view of the AO basis.
            #
            # This is crucial for performance on contracted AO bases, because the contracted
            # DF kernel is a generic Rys implementation and does not benefit from Step-2 native
            # kernels like ssss/psss/ppss/dsss.
            use_step2_hybrid = True
            try:
                import os

                use_step2_hybrid = os.environ.get("CUERI_DF_CONTRACTED_AO_STEP2", "1").strip().lower() not in ("0", "false", "off")
            except Exception:  # pragma: no cover
                use_step2_hybrid = True

            if use_step2_hybrid:
                from .eri_dispatch import run_kernel_batch_spd

                # Warm + cache the expanded view and its DF plan so repeated DF builds (e.g., SCF)
                # don't pay Python planning overhead repeatedly.
                ao_exp = _get_expanded_cart_basis_from_contracted(ao_basis)
                if profile is not None:
                    t0s = cp.cuda.Event()
                    t1s = cp.cuda.Event()
                    t0s.record(s0)
                plan_step2 = _get_df_int3c2e_rys_plan(
                    ao_exp,
                    aux_basis,
                    threads=int(threads),
                    mode="auto",
                    work_small_max=512,
                    work_large_min=200_000,
                    blocks_per_task=4,
                )
                if profile is not None:
                    t1s.record(s0)
                    t1s.synchronize()
                    plan_ms = float(cp.cuda.get_elapsed_time(t0s, t1s))
                    prof = profile.setdefault("df_int3c2e", {})
                    prof["plan_ms"] = float(prof.get("plan_ms", 0.0)) + plan_ms
                    prof["plan_ms_step2"] = float(prof.get("plan_ms_step2", 0.0)) + plan_ms

                for batch, a0_dev, b0_dev, p0_dev, nB in zip(
                    plan_step2.batches,
                    plan_step2.batch_a0_dev,
                    plan_step2.batch_b0_dev,
                    plan_step2.batch_p0_dev,
                    plan_step2.batch_nB,
                ):
                    if int(batch.kernel_class_id) not in _STEP2_BASE_CLASS_IDS:
                        continue

                    start_k = cp.cuda.Event() if profile is not None else None
                    end_k = cp.cuda.Event() if profile is not None else None
                    start_sc = cp.cuda.Event() if profile is not None else None
                    end_sc = cp.cuda.Event() if profile is not None else None
                    if start_k is not None and end_k is not None:
                        start_k.record(s0)

                    tile = run_kernel_batch_spd(
                        batch,
                        dbasis=plan_step2.dbasis,
                        dsp=plan_step2.dsp,
                        pt=plan_step2.pt,
                        stream=stream,
                        threads=int(threads),
                        mode="auto",
                        work_small_max=512,
                        work_large_min=200_000,
                        blocks_per_task=4,
                        profile=dispatch_profile,
                    )

                    if start_k is not None and end_k is not None:
                        end_k.record(s0)
                        end_k.synchronize()
                        k_ms = float(cp.cuda.get_elapsed_time(start_k, end_k))
                        prof = profile.setdefault("df_int3c2e", {})
                        prof["kernel_ms"] = float(prof.get("kernel_ms", 0.0)) + k_ms
                        key = f"{int(batch.kernel_class_id)}|tr={int(batch.transpose)}|nAB={int(tile.shape[1])}|nP={int(tile.shape[2])}|step2_hybrid"
                        row = prof.setdefault("batches", {}).setdefault(key, {"kernel_ms": 0.0, "scatter_ms": 0.0, "ntasks": 0})
                        row["kernel_ms"] = float(row.get("kernel_ms", 0.0)) + k_ms
                        row["ntasks"] = int(row.get("ntasks", 0)) + int(tile.shape[0])

                    if start_sc is not None and end_sc is not None:
                        start_sc.record(s0)

                    nAB = int(tile.shape[1])
                    nP = int(tile.shape[2])
                    _ext.scatter_df_int3c2e_tiles_inplace_device(
                        tile.ravel(),
                        a0_dev,
                        b0_dev,
                        p0_dev,
                        int(plan.nao),
                        int(plan.naux),
                        int(nAB),
                        int(nB),
                        int(nP),
                        X.ravel(),
                        int(threads),
                        int(_stream_ptr(stream)),
                        False,
                    )

                    if start_sc is not None and end_sc is not None:
                        end_sc.record(s0)
                        end_sc.synchronize()
                        sc_ms = float(cp.cuda.get_elapsed_time(start_sc, end_sc))
                        prof = profile.setdefault("df_int3c2e", {})
                        prof["scatter_ms"] = float(prof.get("scatter_ms", 0.0)) + sc_ms
                        key = f"{int(batch.kernel_class_id)}|tr={int(batch.transpose)}|nAB={int(tile.shape[1])}|nP={int(tile.shape[2])}|step2_hybrid"
                        row = prof.setdefault("batches", {}).setdefault(key, {"kernel_ms": 0.0, "scatter_ms": 0.0, "ntasks": 0})
                        row["scatter_ms"] = float(row.get("scatter_ms", 0.0)) + sc_ms

            for gid, cid in enumerate(np.asarray(plan.class_ids, dtype=np.int32).ravel()):
                # Avoid double-computing Step-2-eligible classes when the hybrid path is enabled.
                if use_step2_hybrid and _df_int3c2e_class_id_step2_eligible(int(cid)):
                    continue
                i0 = int(plan.offsets[gid])
                i1 = int(plan.offsets[gid + 1])
                if i1 <= i0:
                    continue
                idx = np.asarray(plan.perm[i0:i1], dtype=np.int32).ravel()
                if idx.size == 0:
                    continue

                la, lb, lc, ld = decode_eri_class_id(int(cid))
                if ld != 0:
                    continue  # should not happen for DF int3c2e

                task_ab = cp.ascontiguousarray(cp.asarray(plan.tasks.task_spAB[idx], dtype=cp.int32))
                task_cd = cp.ascontiguousarray(cp.asarray(plan.tasks.task_spCD[idx], dtype=cp.int32))

                start_k = cp.cuda.Event() if profile is not None else None
                end_k = cp.cuda.Event() if profile is not None else None
                if start_k is not None and end_k is not None:
                    start_k.record(s0)

                fn = (
                    _ext.df_int3c2e_rys_contracted_ctr2_inplace_device
                    if use_ctr2
                    else _ext.df_int3c2e_rys_contracted_inplace_device
                )
                fn(
                    task_ab,
                    task_cd,
                    plan.dsp.sp_A,
                    plan.dsp.sp_B,
                    plan.dsp.sp_pair_start,
                    plan.dsp.sp_npair,
                    plan.dbasis.shell_nprim,
                    plan.dbasis.shell_cx,
                    plan.dbasis.shell_cy,
                    plan.dbasis.shell_cz,
                    plan.pt.pair_eta,
                    plan.pt.pair_Px,
                    plan.pt.pair_Py,
                    plan.pt.pair_Pz,
                    plan.pt.pair_cK,
                    plan.ao_shell_ao_start_dev,
                    plan.ao_shell_nctr_dev,
                    plan.ao_shell_coef_start_dev,
                    plan.ao_prim_coef_dev,
                    plan.aux_shell_ao_start_dev,
                    int(plan.n_shell_ao),
                    int(plan.nao),
                    int(plan.naux),
                    0,  # aux_p0_block
                    int(la),
                    int(lb),
                    int(lc),
                    X.ravel(),
                    int(threads),
                    int(_stream_ptr(stream)),
                    False,
                )

                if start_k is not None and end_k is not None:
                    end_k.record(s0)
                    end_k.synchronize()
                    k_ms = float(cp.cuda.get_elapsed_time(start_k, end_k))
                    prof = profile.setdefault("df_int3c2e", {})
                    prof["kernel_ms"] = float(prof.get("kernel_ms", 0.0)) + k_ms
                    key = f"{int(cid)}|ntasks={int(idx.size)}|contracted_ao_ctr2" if use_ctr2 else f"{int(cid)}|ntasks={int(idx.size)}|contracted_ao"
                    batches = prof.setdefault("batches", {})
                    b = batches.setdefault(key, {})
                    b["kernel_ms"] = float(b.get("kernel_ms", 0.0)) + k_ms
                    b["ntasks"] = int(b.get("ntasks", 0)) + int(idx.size)

            return X

        # Slow path: build a temporary plan for this aux shell block (no caching yet).
        if aux_shell_start == aux_shell_stop:
            return cp.empty((nao, nao, 0), dtype=cp.float64)

        # Combined (AO + aux_block + dummy) basis arrays.
        n_shell_ao = int(ao_basis.shell_cxyz.shape[0])
        n_shell_aux = int(aux_basis.shell_cxyz.shape[0])
        n_shell_aux_blk = int(aux_shell_stop - aux_shell_start)
        dummy_shell = n_shell_ao + n_shell_aux_blk

        ao_prim_n = int(np.asarray(ao_basis.prim_exp, dtype=np.float64).shape[0])
        aux_prim_n = int(np.asarray(aux_basis.prim_exp, dtype=np.float64).shape[0])

        shell_cxyz = np.concatenate(
            [
                np.asarray(ao_basis.shell_cxyz, dtype=np.float64),
                np.asarray(aux_basis.shell_cxyz, dtype=np.float64)[aux_shell_start:aux_shell_stop],
                np.zeros((1, 3), dtype=np.float64),
            ],
            axis=0,
        )
        shell_prim_start = np.concatenate(
            [
                np.asarray(ao_basis.shell_prim_start, dtype=np.int32),
                np.asarray(aux_basis.shell_prim_start, dtype=np.int32)[aux_shell_start:aux_shell_stop] + np.int32(ao_prim_n),
                np.asarray([ao_prim_n + aux_prim_n], dtype=np.int32),
            ]
        )
        shell_nprim = np.concatenate(
            [
                np.asarray(ao_basis.shell_nprim, dtype=np.int32),
                np.asarray(aux_basis.shell_nprim, dtype=np.int32)[aux_shell_start:aux_shell_stop],
                np.asarray([1], dtype=np.int32),
            ]
        )
        prim_exp = np.concatenate(
            [np.asarray(ao_basis.prim_exp, dtype=np.float64), np.asarray(aux_basis.prim_exp, dtype=np.float64), np.asarray([0.0])]
        )
        prim_coef = np.concatenate(
            [
                np.ones((ao_prim_n,), dtype=np.float64),
                np.asarray(aux_basis.prim_coef, dtype=np.float64),
                np.asarray([1.0], dtype=np.float64),
            ]
        )
        shell_l = np.concatenate([ao_l, aux_l[aux_shell_start:aux_shell_stop], np.asarray([0], dtype=np.int32)])

        basis_like = SimpleNamespace(
            shell_cxyz=shell_cxyz,
            shell_prim_start=shell_prim_start,
            shell_nprim=shell_nprim,
            prim_exp=prim_exp,
            prim_coef=prim_coef,
        )

        # ShellPairs: AO pairs (full) + (P, dummy) for aux shells in block (local indexing).
        sp_ao = build_shell_pairs_l_order(ao_basis)
        nsp_ao = int(sp_ao.sp_A.shape[0])

        aux_shell_idx = np.arange(n_shell_aux_blk, dtype=np.int32)
        sp_aux_A = (aux_shell_idx + np.int32(n_shell_ao)).astype(np.int32, copy=False)
        sp_aux_B = np.full((int(aux_shell_idx.size),), int(dummy_shell), dtype=np.int32)
        sp_aux_npair = np.asarray(aux_basis.shell_nprim, dtype=np.int32)[aux_shell_start:aux_shell_stop].astype(np.int32, copy=False)

        sp_A = np.concatenate([np.asarray(sp_ao.sp_A, dtype=np.int32), sp_aux_A])
        sp_B = np.concatenate([np.asarray(sp_ao.sp_B, dtype=np.int32), sp_aux_B])
        sp_npair = np.concatenate([np.asarray(sp_ao.sp_npair, dtype=np.int32), sp_aux_npair])
        sp_pair_start = np.empty((int(sp_npair.shape[0]) + 1,), dtype=np.int32)
        sp_pair_start[0] = 0
        sp_pair_start[1:] = np.cumsum(sp_npair, dtype=np.int32)
        sp_all = ShellPairs(sp_A=sp_A, sp_B=sp_B, sp_npair=sp_npair, sp_pair_start=sp_pair_start)

        dbasis = to_device_basis_ss(basis_like)
        dsp = to_device_shell_pairs(sp_all)
        pt = build_pair_tables_ss_device(dbasis, dsp, stream=stream, threads=int(threads))

        # Tasks: all (AO shell pair, aux shell) combinations for this block.
        task_ab = np.tile(np.arange(nsp_ao, dtype=np.int32), int(n_shell_aux_blk))
        task_cd = np.repeat((np.int32(nsp_ao) + np.arange(n_shell_aux_blk, dtype=np.int32)), int(nsp_ao))
        tasks = TaskList(task_spAB=task_ab, task_spCD=task_cd)
        tasks = with_task_class_id(tasks, sp_all, shell_l)

        perm, class_ids, offsets = group_tasks_by_class(tasks.task_class_id)

        aux_start = np.asarray(aux_basis.shell_ao_start, dtype=np.int32)
        p0_block = int(aux_start[aux_shell_start])
        p1_block = int(aux_start[aux_shell_stop - 1]) + int(ncart(int(aux_l[aux_shell_stop - 1])))
        naux_blk = int(p1_block - p0_block)

        X = cp.zeros((nao, nao, naux_blk), dtype=cp.float64)

        use_ctr2 = int(ao_nctr.max()) <= 2

        ao_shell_ao_start_dev = cp.ascontiguousarray(cp.asarray(ao_start, dtype=cp.int32))
        ao_shell_nctr_dev = cp.ascontiguousarray(cp.asarray(ao_nctr, dtype=cp.int32))
        ao_shell_coef_start_dev = cp.ascontiguousarray(cp.asarray(np.asarray(ao_basis.shell_coef_start, dtype=np.int32), dtype=cp.int32))
        ao_prim_coef_dev = cp.ascontiguousarray(cp.asarray(np.asarray(ao_basis.prim_coef_flat, dtype=np.float64), dtype=cp.float64))
        aux_shell_ao_start_dev = cp.ascontiguousarray(cp.asarray(aux_start[aux_shell_start:aux_shell_stop], dtype=cp.int32))

        s0 = cp.cuda.get_current_stream()
        for gid, cid in enumerate(np.asarray(class_ids, dtype=np.int32).ravel()):
            i0 = int(offsets[gid])
            i1 = int(offsets[gid + 1])
            if i1 <= i0:
                continue
            idx = np.asarray(perm[i0:i1], dtype=np.int32).ravel()
            if idx.size == 0:
                continue
            la, lb, lc, ld = decode_eri_class_id(int(cid))
            if ld != 0:
                continue
            task_ab = cp.ascontiguousarray(cp.asarray(tasks.task_spAB[idx], dtype=cp.int32))
            task_cd = cp.ascontiguousarray(cp.asarray(tasks.task_spCD[idx], dtype=cp.int32))

            fn = (
                _ext.df_int3c2e_rys_contracted_ctr2_inplace_device
                if use_ctr2
                else _ext.df_int3c2e_rys_contracted_inplace_device
            )
            fn(
                task_ab,
                task_cd,
                dsp.sp_A,
                dsp.sp_B,
                dsp.sp_pair_start,
                dsp.sp_npair,
                dbasis.shell_nprim,
                dbasis.shell_cx,
                dbasis.shell_cy,
                dbasis.shell_cz,
                pt.pair_eta,
                pt.pair_Px,
                pt.pair_Py,
                pt.pair_Pz,
                pt.pair_cK,
                ao_shell_ao_start_dev,
                ao_shell_nctr_dev,
                ao_shell_coef_start_dev,
                ao_prim_coef_dev,
                aux_shell_ao_start_dev,
                int(n_shell_ao),
                int(nao),
                int(naux_blk),
                int(p0_block),
                int(la),
                int(lb),
                int(lc),
                X.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )

        return X


def df_int3c2e_rys_device_block(
    ao_basis,
    aux_basis,
    *,
    aux_shell_start: int,
    aux_shell_stop: int,
    stream=None,
    threads: int = 256,
    mode: str = "auto",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 4,
    ao_contract_mode: str = "auto",
    ao_rep: str = "cart",
    profile: dict | None = None,
):
    """Compute 3c2e integrals X(μ,ν,P) = (μν|P) on GPU for AO+aux bases (general-l, reference-oriented).

    Uses the Step-2 4c2e evaluator via the identity:
      (μν|P) == (μν | P*1)
    where "1" is modeled as a dummy s-shell with one primitive exp=0, coef=1.

    Current limitation
    - requires `ao_basis`/`aux_basis` to provide `shell_ao_start` (i.e., `BasisCartSoA`)
    - CUDA backend supports `l<=CUDA_MAX_L` per shell (nroots<=CUDA_MAX_NROOTS)

    Parameters
    - ao_rep: 'cart' or 'sph'. When 'sph', scatters directly into spherical AO space
      (real spherical harmonics) without materializing a full Cartesian X tensor.
    """

    import cupy as cp

    _require_cuda_ext()
    if profile is not None:
        prof = profile.setdefault("df_int3c2e", {})
        prof.setdefault("ao_contract_mode", str(ao_contract_mode).lower().strip())
        prof.setdefault("ao_contract_path", "expanded")
        prof.setdefault("ao_contract_fallback_reason", None)
        prof.setdefault("plan_ms", 0.0)
        prof.setdefault("kernel_ms", 0.0)
        prof.setdefault("scatter_ms", 0.0)
        prof.setdefault("batches", {})
    dispatch_profile = None
    if profile is not None:
        dispatch_profile = profile.setdefault("df_int3c2e", {}).setdefault("kernel_dispatch", {})
    ao_contract_mode = str(ao_contract_mode).lower().strip()
    if ao_contract_mode not in ("auto", "expanded", "native_contracted"):
        raise ValueError("ao_contract_mode must be one of: 'auto', 'expanded', 'native_contracted'")

    ao_rep_s = str(ao_rep).lower().strip()
    if ao_rep_s not in ("cart", "sph"):
        raise ValueError("ao_rep must be one of: 'cart', 'sph'")
    if ao_rep_s == "sph" and (_ext is None or not hasattr(_ext, "scatter_df_int3c2e_tiles_cart_to_sph_inplace_device")):
        raise RuntimeError(
            "ao_rep='sph' requires a cuERI CUDA extension with spherical DF scatter support; "
            "rebuild via `python -m asuka.cueri.build_cuda_ext`"
        )

    if profile is not None:
        prof = profile.setdefault("df_int3c2e", {})
        prof.setdefault("ao_rep", str(ao_rep_s))

    is_contracted_ao = (
        hasattr(ao_basis, "shell_nctr") and hasattr(ao_basis, "shell_coef_start") and hasattr(ao_basis, "prim_coef_flat")
    )
    if ao_contract_mode == "native_contracted" and not is_contracted_ao:
        raise TypeError("ao_contract_mode='native_contracted' requires a contracted AO basis")

    # Contracted AO path: either explicitly requested, or auto-selected when supported.
    if is_contracted_ao and ao_contract_mode in ("auto", "native_contracted"):
        use_native = False
        fallback_reason = None
        if ao_contract_mode == "native_contracted":
            use_native = True
        else:
            # Auto policy: prefer native contracted path when kernel constraints are met.
            ao_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
            aux_l = np.asarray(aux_basis.shell_l, dtype=np.int32).ravel()
            ao_nctr = np.asarray(ao_basis.shell_nctr, dtype=np.int32).ravel()
            if hasattr(aux_basis, "shell_nctr") or hasattr(aux_basis, "prim_coef_flat"):
                fallback_reason = "unsupported_aux_contracted"
            elif ao_l.size == 0 or aux_l.size == 0:
                # Degenerate cases are handled by native path too.
                use_native = True
            elif int(ao_l.max()) > CUDA_MAX_L or int(aux_l.max()) > CUDA_MAX_L:
                fallback_reason = "lmax_limit"
            elif int(ao_nctr.max()) > CUDA_MAX_L:
                fallback_reason = "nctr_limit"
            else:
                use_native = True

        if ao_rep_s == "sph":
            if ao_contract_mode == "native_contracted":
                raise NotImplementedError("ao_rep='sph' does not support ao_contract_mode='native_contracted' (use 'expanded')")
            if use_native:
                use_native = False
                fallback_reason = "ao_rep_sph_requires_expanded"

        if use_native:
            if profile is not None:
                prof = profile.setdefault("df_int3c2e", {})
                prof["ao_contract_mode"] = str(ao_contract_mode)
                prof["ao_contract_path"] = "native_contracted"
            return df_int3c2e_rys_contracted_ao_device_block(
                ao_basis,
                aux_basis,
                aux_shell_start=aux_shell_start,
                aux_shell_stop=aux_shell_stop,
                stream=stream,
                threads=threads,
                profile=profile,
            )

        if profile is not None:
            prof = profile.setdefault("df_int3c2e", {})
            prof["ao_contract_mode"] = "auto"
            prof["ao_contract_path"] = "expanded"
            if fallback_reason is not None:
                prof["ao_contract_fallback_reason"] = str(fallback_reason)

        # Auto fallback: expand contracted AO shells and run the standard Step-2 path.
        ao_basis = _get_expanded_cart_basis_from_contracted(ao_basis)

    if is_contracted_ao and ao_contract_mode == "expanded":
        ao_basis = _get_expanded_cart_basis_from_contracted(ao_basis)

    if profile is not None:
        prof = profile.setdefault("df_int3c2e", {})
        prof.setdefault("ao_contract_mode", str(ao_contract_mode))
        if is_contracted_ao:
            prof.setdefault("ao_contract_path", "expanded" if ao_contract_mode == "expanded" else "native_contracted")
        else:
            prof.setdefault("ao_contract_path", "expanded")

    with stream_ctx(stream):
        if not hasattr(ao_basis, "shell_ao_start") or not hasattr(aux_basis, "shell_ao_start"):
            raise TypeError("ao_basis/aux_basis must provide shell_ao_start (use packed cartesian bases like BasisCartSoA)")

        ao_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
        aux_l = np.asarray(aux_basis.shell_l, dtype=np.int32).ravel()

        aux_shell_start = int(aux_shell_start)
        aux_shell_stop = int(aux_shell_stop)
        if aux_shell_start < 0 or aux_shell_stop < aux_shell_start or aux_shell_stop > int(aux_l.size):
            raise ValueError("aux_shell_start/aux_shell_stop out of range")

        if ao_l.size == 0 or aux_l.size == 0 or aux_shell_start == aux_shell_stop:
            if ao_rep_s == "cart":
                ao_start = np.asarray(getattr(ao_basis, "shell_ao_start", np.asarray([0], dtype=np.int32)), dtype=np.int32)
                ao_nfunc = (
                    np.asarray([ncart(int(l)) for l in ao_l], dtype=np.int32)
                    if ao_l.size
                    else np.asarray([0], dtype=np.int32)
                )
                nao_out = int(np.max(ao_start + ao_nfunc)) if ao_l.size else 0
            else:
                # Expanded cart shells -> real spherical harmonics have nsph(l) = 2*l+1.
                nao_out = int(sum(int(int(l) * 2 + 1) for l in ao_l)) if ao_l.size else 0
            return cp.empty((nao_out, nao_out, 0), dtype=cp.float64)

        if int(ao_l.max()) > CUDA_MAX_L or int(aux_l.max()) > CUDA_MAX_L:
            raise NotImplementedError(
                f"df_int3c2e_rys_device currently supports only l<={CUDA_MAX_L} for AO and aux shells"
            )

        mode = mode.lower().strip()
        if mode not in ("block", "warp", "multiblock", "auto"):
            raise ValueError("mode must be one of: 'block', 'warp', 'multiblock', 'auto'")

        # Fast path: cache the full-aux plan to avoid Python planning + repeated H2D index transfers
        # for microbenchmarks and materialized 3c2e builds.
        if aux_shell_start == 0 and aux_shell_stop == int(aux_l.size):
            s0 = cp.cuda.get_current_stream()
            if profile is not None:
                t0 = cp.cuda.Event()
                t1 = cp.cuda.Event()
                t0.record(s0)
            plan = _get_df_int3c2e_rys_plan(
                ao_basis,
                aux_basis,
                threads=int(threads),
                mode=str(mode),
                work_small_max=int(work_small_max),
                work_large_min=int(work_large_min),
                blocks_per_task=int(blocks_per_task),
            )
            if profile is not None:
                t1.record(s0)
                t1.synchronize()
                prof = profile.setdefault("df_int3c2e", {})
                prof["plan_ms"] = float(cp.cuda.get_elapsed_time(t0, t1))

            nao_out = int(plan.nao_sph) if ao_rep_s == "sph" else int(plan.nao)
            X = cp.zeros((nao_out, nao_out, int(plan.naux)), dtype=cp.float64)
            from .eri_dispatch import run_kernel_batch_spd

            for batch, a0_dev, b0_dev, a0_sph_dev, b0_sph_dev, p0_dev, nB, la, lb in zip(
                plan.batches,
                plan.batch_a0_dev,
                plan.batch_b0_dev,
                plan.batch_a0_sph_dev,
                plan.batch_b0_sph_dev,
                plan.batch_p0_dev,
                plan.batch_nB,
                plan.batch_la,
                plan.batch_lb,
            ):
                start_k = cp.cuda.Event() if profile is not None else None
                end_k = cp.cuda.Event() if profile is not None else None
                start_sc = cp.cuda.Event() if profile is not None else None
                end_sc = cp.cuda.Event() if profile is not None else None
                if start_k is not None and end_k is not None:
                    start_k.record(s0)
                tile = run_kernel_batch_spd(
                    batch,
                    dbasis=plan.dbasis,
                    dsp=plan.dsp,
                    pt=plan.pt,
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
                    prof = profile.setdefault("df_int3c2e", {})
                    prof["kernel_ms"] = float(prof.get("kernel_ms", 0.0)) + k_ms
                    key = f"{int(batch.kernel_class_id)}|tr={int(batch.transpose)}|nAB={int(tile.shape[1])}|nP={int(tile.shape[2])}"
                    row = prof.setdefault("batches", {}).setdefault(key, {"kernel_ms": 0.0, "scatter_ms": 0.0, "ntasks": 0})
                    row["kernel_ms"] = float(row.get("kernel_ms", 0.0)) + k_ms
                    row["ntasks"] = int(row.get("ntasks", 0)) + int(tile.shape[0])
                nAB = int(tile.shape[1])
                nP = int(tile.shape[2])
                if start_sc is not None and end_sc is not None:
                    start_sc.record(s0)
                if ao_rep_s == "cart":
                    _ext.scatter_df_int3c2e_tiles_inplace_device(
                        tile.ravel(),
                        a0_dev,
                        b0_dev,
                        p0_dev,
                        int(plan.nao),
                        int(plan.naux),
                        int(nAB),
                        int(nB),
                        int(nP),
                        X.ravel(),
                        int(threads),
                        int(_stream_ptr(stream)),
                        False,
                    )
                else:
                    _ext.scatter_df_int3c2e_tiles_cart_to_sph_inplace_device(
                        tile.ravel(),
                        a0_sph_dev,
                        b0_sph_dev,
                        p0_dev,
                        int(plan.nao_sph),
                        int(plan.naux),
                        int(nAB),
                        int(nB),
                        int(nP),
                        int(la),
                        int(lb),
                        X.ravel(),
                        int(threads),
                        int(_stream_ptr(stream)),
                        False,
                    )
                if start_sc is not None and end_sc is not None:
                    end_sc.record(s0)
                    end_sc.synchronize()
                    sc_ms = float(cp.cuda.get_elapsed_time(start_sc, end_sc))
                    prof = profile.setdefault("df_int3c2e", {})
                    prof["scatter_ms"] = float(prof.get("scatter_ms", 0.0)) + sc_ms
                    key = f"{int(batch.kernel_class_id)}|tr={int(batch.transpose)}|nAB={int(tile.shape[1])}|nP={int(tile.shape[2])}"
                    row = prof.setdefault("batches", {}).setdefault(key, {"kernel_ms": 0.0, "scatter_ms": 0.0, "ntasks": 0})
                    row["scatter_ms"] = float(row.get("scatter_ms", 0.0)) + sc_ms

            return X

        # Combined (AO + aux + dummy) basis arrays.
        n_shell_ao = int(ao_basis.shell_cxyz.shape[0])
        n_shell_aux = int(aux_basis.shell_cxyz.shape[0])
        dummy_shell = n_shell_ao + n_shell_aux

        ao_prim_n = int(ao_basis.prim_exp.shape[0])
        aux_prim_n = int(aux_basis.prim_exp.shape[0])

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
            [
                np.asarray(ao_basis.prim_exp, dtype=np.float64),
                np.asarray(aux_basis.prim_exp, dtype=np.float64),
                np.asarray([0.0]),
            ]
        )
        prim_coef = np.concatenate(
            [
                np.asarray(ao_basis.prim_coef, dtype=np.float64),
                np.asarray(aux_basis.prim_coef, dtype=np.float64),
                np.asarray([1.0]),
            ]
        )
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

        # - Aux side: (P, dummy) for each aux shell P in the requested shell block.
        aux_shell_idx = np.arange(aux_shell_start, aux_shell_stop, dtype=np.int32)
        sp_aux_A = (aux_shell_idx + np.int32(n_shell_ao)).astype(np.int32, copy=False)
        sp_aux_B = np.full((int(aux_shell_idx.size),), int(dummy_shell), dtype=np.int32)
        sp_aux_npair = np.asarray(np.asarray(aux_basis.shell_nprim, dtype=np.int32)[aux_shell_start:aux_shell_stop], dtype=np.int32)

        sp_A = np.concatenate([np.asarray(sp_ao.sp_A, dtype=np.int32), sp_aux_A])
        sp_B = np.concatenate([np.asarray(sp_ao.sp_B, dtype=np.int32), sp_aux_B])
        sp_npair = np.concatenate([np.asarray(sp_ao.sp_npair, dtype=np.int32), sp_aux_npair])
        sp_pair_start = np.empty((int(sp_npair.shape[0]) + 1,), dtype=np.int32)
        sp_pair_start[0] = 0
        sp_pair_start[1:] = np.cumsum(sp_npair, dtype=np.int32)
        sp_all = ShellPairs(sp_A=sp_A, sp_B=sp_B, sp_npair=sp_npair, sp_pair_start=sp_pair_start)

        dbasis = to_device_basis_ss(basis_like)
        dsp = to_device_shell_pairs(sp_all)
        pt = build_pair_tables_ss_device(dbasis, dsp, stream=stream, threads=threads)

        # Tasks: all (AO shell pair, aux shell) combinations.
        n_aux_shell_blk = int(aux_shell_stop - aux_shell_start)
        task_ab = np.repeat(np.arange(nsp_ao, dtype=np.int32), n_aux_shell_blk)
        task_cd = (nsp_ao + np.tile(np.arange(n_aux_shell_blk, dtype=np.int32), nsp_ao)).astype(np.int32, copy=False)
        tasks = TaskList(task_spAB=task_ab, task_spCD=task_cd)
        tasks = with_task_class_id(tasks, sp_all, shell_l)

        ao_start = np.asarray(ao_basis.shell_ao_start, dtype=np.int32)
        ao_nfunc = np.asarray([ncart(int(l)) for l in ao_l], dtype=np.int32)
        nao_cart = int(np.max(ao_start + ao_nfunc))
        shell_ao_start_sph = None
        if ao_rep_s == "sph":
            shell_ao_start_sph = np.empty((int(n_shell_ao),), dtype=np.int32)
            cursor = 0
            for sh in range(int(n_shell_ao)):
                shell_ao_start_sph[sh] = np.int32(cursor)
                cursor += int(int(ao_l[sh]) * 2 + 1)
            nao_out = int(cursor)
        else:
            nao_out = int(nao_cart)

        aux_start = np.asarray(aux_basis.shell_ao_start, dtype=np.int32)
        p0_block = int(aux_start[aux_shell_start])
        p1_block = int(aux_start[aux_shell_stop - 1]) + int(ncart(int(aux_l[aux_shell_stop - 1])))
        naux = int(p1_block - p0_block)

        X = cp.zeros((int(nao_out), int(nao_out), int(naux)), dtype=cp.float64)

        from .eri_dispatch import plan_kernel_batches_spd, run_kernel_batch_spd

        batches = plan_kernel_batches_spd(tasks, shell_pairs=sp_all, shell_l=shell_l)
        for batch in batches:
            tile = run_kernel_batch_spd(
                batch,
                dbasis=dbasis,
                dsp=dsp,
                pt=pt,
                stream=stream,
                threads=threads,
                mode=mode,
                work_small_max=work_small_max,
                work_large_min=work_large_min,
                blocks_per_task=blocks_per_task,
            )

            idx = np.asarray(batch.task_idx, dtype=np.int32).ravel()
            if idx.size == 0:
                continue
            spab = np.asarray(tasks.task_spAB[idx], dtype=np.int32).ravel()
            spcd = np.asarray(tasks.task_spCD[idx], dtype=np.int32).ravel()

            A = np.asarray(sp_all.sp_A[spab], dtype=np.int32).ravel()
            B = np.asarray(sp_all.sp_B[spab], dtype=np.int32).ravel()
            P = (np.asarray(sp_all.sp_A[spcd], dtype=np.int32).ravel() - np.int32(n_shell_ao)).astype(np.int32, copy=False)

            p0 = (np.asarray(aux_basis.shell_ao_start[P], dtype=np.int32).ravel() - np.int32(p0_block)).astype(np.int32, copy=False)

            p0_dev = cp.ascontiguousarray(cp.asarray(p0, dtype=cp.int32))

            nAB = int(tile.shape[1])
            nP = int(tile.shape[2])
            lb = int(ao_l[int(B[0])])
            nB = int(ncart(lb))
            if ao_rep_s == "cart":
                a0 = np.asarray(ao_basis.shell_ao_start[A], dtype=np.int32).ravel()
                b0 = np.asarray(ao_basis.shell_ao_start[B], dtype=np.int32).ravel()
                a0_dev = cp.ascontiguousarray(cp.asarray(a0, dtype=cp.int32))
                b0_dev = cp.ascontiguousarray(cp.asarray(b0, dtype=cp.int32))
                _ext.scatter_df_int3c2e_tiles_inplace_device(
                    tile.ravel(),
                    a0_dev,
                    b0_dev,
                    p0_dev,
                    int(nao_cart),
                    int(naux),
                    int(nAB),
                    int(nB),
                    int(nP),
                    X.ravel(),
                    int(threads),
                    int(_stream_ptr(stream)),
                    False,
                )
            else:
                if shell_ao_start_sph is None:  # pragma: no cover
                    raise RuntimeError("internal error: shell_ao_start_sph is None for ao_rep='sph'")
                a0_sph = np.asarray(shell_ao_start_sph[A], dtype=np.int32).ravel()
                b0_sph = np.asarray(shell_ao_start_sph[B], dtype=np.int32).ravel()
                a0_sph_dev = cp.ascontiguousarray(cp.asarray(a0_sph, dtype=cp.int32))
                b0_sph_dev = cp.ascontiguousarray(cp.asarray(b0_sph, dtype=cp.int32))
                la = int(ao_l[int(A[0])])
                _ext.scatter_df_int3c2e_tiles_cart_to_sph_inplace_device(
                    tile.ravel(),
                    a0_sph_dev,
                    b0_sph_dev,
                    p0_dev,
                    int(nao_out),
                    int(naux),
                    int(nAB),
                    int(nB),
                    int(nP),
                    int(la),
                    int(lb),
                    X.ravel(),
                    int(threads),
                    int(_stream_ptr(stream)),
                    False,
                )

        return X


def eri_psss_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    """Evaluate contracted (ps|ss) shell-quartet tiles (Cartesian) for tasks of class (1,0,0,0).

    Output layout: a flat device array of length `ntasks*3`, with `[x,y,z]` for each task.
    """

    import cupy as cp

    _require_cuda_ext()
    with stream_ctx(stream):
        task_ab = cp.asarray(tasks.task_spAB, dtype=cp.int32)
        task_cd = cp.asarray(tasks.task_spCD, dtype=cp.int32)
        task_ab = cp.ascontiguousarray(task_ab)
        task_cd = cp.ascontiguousarray(task_cd)

        out = cp.empty((tasks.ntasks, 3), dtype=cp.float64)
        if tasks.ntasks == 0:
            return out.ravel()

        mode = mode.lower().strip()
        if mode not in ("block", "warp", "multiblock", "auto"):
            raise ValueError("mode must be one of: 'block', 'warp', 'multiblock', 'auto'")

        if mode == "block":
            _ext.eri_psss_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        if mode == "warp":
            _ext.eri_psss_warp_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        if mode == "multiblock":
            if blocks_per_task <= 0:
                raise ValueError("blocks_per_task must be > 0")
            partial = cp.empty((tasks.ntasks * int(blocks_per_task) * 3,), dtype=cp.float64)
            _ext.eri_psss_multiblock_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                partial,
                int(blocks_per_task),
                out.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        # mode == "auto": bin by a crude work proxy.
        if work_small_max < 0 or work_large_min < 0:
            raise ValueError("work_small_max/work_large_min must be >= 0")
        if work_small_max >= work_large_min:
            raise ValueError("work_small_max must be < work_large_min")

        work = shell_pairs.sp_npair[task_ab].astype(cp.int64) * shell_pairs.sp_npair[task_cd].astype(cp.int64)
        small_mask = work <= int(work_small_max)
        large_mask = work >= int(work_large_min)
        med_mask = ~(small_mask | large_mask)

        small_idx = cp.nonzero(small_mask)[0]
        med_idx = cp.nonzero(med_mask)[0]
        large_idx = cp.nonzero(large_mask)[0]

        if int(small_idx.size) > 0:
            ab_small = cp.ascontiguousarray(task_ab[small_idx])
            cd_small = cp.ascontiguousarray(task_cd[small_idx])
            out_small = cp.empty((int(ab_small.shape[0]), 3), dtype=cp.float64)
            _ext.eri_psss_warp_inplace_device(
                ab_small,
                cd_small,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out_small.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            out[small_idx] = out_small

        if int(med_idx.size) > 0:
            ab_med = cp.ascontiguousarray(task_ab[med_idx])
            cd_med = cp.ascontiguousarray(task_cd[med_idx])
            out_med = cp.empty((int(ab_med.shape[0]), 3), dtype=cp.float64)
            _ext.eri_psss_inplace_device(
                ab_med,
                cd_med,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out_med.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            out[med_idx] = out_med

        if int(large_idx.size) > 0:
            if blocks_per_task <= 0:
                raise ValueError("blocks_per_task must be > 0")
            ab_large = cp.ascontiguousarray(task_ab[large_idx])
            cd_large = cp.ascontiguousarray(task_cd[large_idx])
            out_large = cp.empty((int(ab_large.shape[0]), 3), dtype=cp.float64)
            partial = cp.empty((int(ab_large.shape[0]) * int(blocks_per_task) * 3,), dtype=cp.float64)
            _ext.eri_psss_multiblock_inplace_device(
                ab_large,
                cd_large,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                partial,
                int(blocks_per_task),
                out_large.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            out[large_idx] = out_large
        return out.ravel()


def eri_ppss_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    """Evaluate contracted (pp|ss) shell-quartet tiles (Cartesian) for tasks of class (1,1,0,0).

    Output layout: a flat device array of length `ntasks*9`, with `[A(x,y,z), B(x,y,z)]` row-major
    (A-major, B-minor) for each task.
    """

    import cupy as cp

    _require_cuda_ext()
    with stream_ctx(stream):
        task_ab = cp.asarray(tasks.task_spAB, dtype=cp.int32)
        task_cd = cp.asarray(tasks.task_spCD, dtype=cp.int32)
        task_ab = cp.ascontiguousarray(task_ab)
        task_cd = cp.ascontiguousarray(task_cd)

        out = cp.empty((tasks.ntasks, 9), dtype=cp.float64)
        if tasks.ntasks == 0:
            return out.ravel()

        mode = mode.lower().strip()
        if mode not in ("block", "warp", "multiblock", "auto"):
            raise ValueError("mode must be one of: 'block', 'warp', 'multiblock', 'auto'")

        if mode == "block":
            _ext.eri_ppss_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        if mode == "warp":
            _ext.eri_ppss_warp_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        if mode == "multiblock":
            if blocks_per_task <= 0:
                raise ValueError("blocks_per_task must be > 0")
            partial = cp.empty((tasks.ntasks * int(blocks_per_task) * 9,), dtype=cp.float64)
            _ext.eri_ppss_multiblock_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                partial,
                int(blocks_per_task),
                out.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        # mode == "auto": bin by a crude work proxy.
        if work_small_max < 0 or work_large_min < 0:
            raise ValueError("work_small_max/work_large_min must be >= 0")
        if work_small_max >= work_large_min:
            raise ValueError("work_small_max must be < work_large_min")

        work = shell_pairs.sp_npair[task_ab].astype(cp.int64) * shell_pairs.sp_npair[task_cd].astype(cp.int64)
        small_mask = work <= int(work_small_max)
        large_mask = work >= int(work_large_min)
        med_mask = ~(small_mask | large_mask)

        small_idx = cp.nonzero(small_mask)[0]
        med_idx = cp.nonzero(med_mask)[0]
        large_idx = cp.nonzero(large_mask)[0]

        if int(small_idx.size) > 0:
            ab_small = cp.ascontiguousarray(task_ab[small_idx])
            cd_small = cp.ascontiguousarray(task_cd[small_idx])
            out_small = cp.empty((int(ab_small.shape[0]), 9), dtype=cp.float64)
            _ext.eri_ppss_warp_inplace_device(
                ab_small,
                cd_small,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out_small.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            out[small_idx] = out_small

        if int(med_idx.size) > 0:
            ab_med = cp.ascontiguousarray(task_ab[med_idx])
            cd_med = cp.ascontiguousarray(task_cd[med_idx])
            out_med = cp.empty((int(ab_med.shape[0]), 9), dtype=cp.float64)
            _ext.eri_ppss_inplace_device(
                ab_med,
                cd_med,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out_med.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            out[med_idx] = out_med

        if int(large_idx.size) > 0:
            if blocks_per_task <= 0:
                raise ValueError("blocks_per_task must be > 0")
            ab_large = cp.ascontiguousarray(task_ab[large_idx])
            cd_large = cp.ascontiguousarray(task_cd[large_idx])
            out_large = cp.empty((int(ab_large.shape[0]), 9), dtype=cp.float64)
            partial = cp.empty((int(ab_large.shape[0]) * int(blocks_per_task) * 9,), dtype=cp.float64)
            _ext.eri_ppss_multiblock_inplace_device(
                ab_large,
                cd_large,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                partial,
                int(blocks_per_task),
                out_large.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            out[large_idx] = out_large
        return out.ravel()


def eri_psps_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    """Evaluate contracted (ps|ps) shell-quartet tiles (Cartesian) for tasks of class (1,0,1,0).

    Output layout: a flat device array of length `ntasks*9`, with `[A(x,y,z), C(x,y,z)]` row-major
    (A-major, C-minor) for each task.
    """

    import cupy as cp

    _require_cuda_ext()
    with stream_ctx(stream):
        task_ab = cp.asarray(tasks.task_spAB, dtype=cp.int32)
        task_cd = cp.asarray(tasks.task_spCD, dtype=cp.int32)
        task_ab = cp.ascontiguousarray(task_ab)
        task_cd = cp.ascontiguousarray(task_cd)

        out = cp.empty((tasks.ntasks, 9), dtype=cp.float64)
        if tasks.ntasks == 0:
            return out.ravel()

        mode = mode.lower().strip()
        if mode not in ("block", "warp", "multiblock", "auto"):
            raise ValueError("mode must be one of: 'block', 'warp', 'multiblock', 'auto'")

        if mode == "block":
            _ext.eri_psps_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        if mode == "warp":
            _ext.eri_psps_warp_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        if mode == "multiblock":
            if blocks_per_task <= 0:
                raise ValueError("blocks_per_task must be > 0")
            partial = cp.empty((tasks.ntasks * int(blocks_per_task) * 9,), dtype=cp.float64)
            _ext.eri_psps_multiblock_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                partial,
                int(blocks_per_task),
                out.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        # mode == "auto": bin by a crude work proxy.
        if work_small_max < 0 or work_large_min < 0:
            raise ValueError("work_small_max/work_large_min must be >= 0")
        if work_small_max >= work_large_min:
            raise ValueError("work_small_max must be < work_large_min")

        work = shell_pairs.sp_npair[task_ab].astype(cp.int64) * shell_pairs.sp_npair[task_cd].astype(cp.int64)
        small_mask = work <= int(work_small_max)
        large_mask = work >= int(work_large_min)
        med_mask = ~(small_mask | large_mask)

        small_idx = cp.nonzero(small_mask)[0]
        med_idx = cp.nonzero(med_mask)[0]
        large_idx = cp.nonzero(large_mask)[0]

        if int(small_idx.size) > 0:
            ab_small = cp.ascontiguousarray(task_ab[small_idx])
            cd_small = cp.ascontiguousarray(task_cd[small_idx])
            out_small = cp.empty((int(ab_small.shape[0]), 9), dtype=cp.float64)
            _ext.eri_psps_warp_inplace_device(
                ab_small,
                cd_small,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out_small.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            out[small_idx] = out_small

        if int(med_idx.size) > 0:
            ab_med = cp.ascontiguousarray(task_ab[med_idx])
            cd_med = cp.ascontiguousarray(task_cd[med_idx])
            out_med = cp.empty((int(ab_med.shape[0]), 9), dtype=cp.float64)
            _ext.eri_psps_inplace_device(
                ab_med,
                cd_med,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out_med.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            out[med_idx] = out_med

        if int(large_idx.size) > 0:
            if blocks_per_task <= 0:
                raise ValueError("blocks_per_task must be > 0")
            ab_large = cp.ascontiguousarray(task_ab[large_idx])
            cd_large = cp.ascontiguousarray(task_cd[large_idx])
            out_large = cp.empty((int(ab_large.shape[0]), 9), dtype=cp.float64)
            partial = cp.empty((int(ab_large.shape[0]) * int(blocks_per_task) * 9,), dtype=cp.float64)
            _ext.eri_psps_multiblock_inplace_device(
                ab_large,
                cd_large,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                partial,
                int(blocks_per_task),
                out_large.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            out[large_idx] = out_large
        return out.ravel()


def eri_ppps_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    """Evaluate contracted (pp|ps) shell-quartet tiles (Cartesian) for tasks of class (1,1,1,0).

    Output layout: a flat device array of length `ntasks*27`, with `AB=9` and `CD=3` stored row-major
    as `tile[ab, cd]` where `ab = ia*3 + ib` (A-major, B-minor), `cd = ic` (C-major, D=s).
    """

    import cupy as cp

    _require_cuda_ext()
    with stream_ctx(stream):
        task_ab = cp.asarray(tasks.task_spAB, dtype=cp.int32)
        task_cd = cp.asarray(tasks.task_spCD, dtype=cp.int32)
        task_ab = cp.ascontiguousarray(task_ab)
        task_cd = cp.ascontiguousarray(task_cd)

        out = cp.empty((tasks.ntasks, 27), dtype=cp.float64)
        if tasks.ntasks == 0:
            return out.ravel()

        mode = mode.lower().strip()
        if mode not in ("block", "warp", "multiblock", "auto"):
            raise ValueError("mode must be one of: 'block', 'warp', 'multiblock', 'auto'")

        if mode == "block":
            _ext.eri_ppps_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        if mode == "warp":
            _ext.eri_ppps_warp_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        if mode == "multiblock":
            if blocks_per_task <= 0:
                raise ValueError("blocks_per_task must be > 0")
            partial = cp.empty((tasks.ntasks * int(blocks_per_task) * 27,), dtype=cp.float64)
            _ext.eri_ppps_multiblock_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                partial,
                int(blocks_per_task),
                out.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        # mode == "auto": bin by a crude work proxy.
        if work_small_max < 0 or work_large_min < 0:
            raise ValueError("work_small_max/work_large_min must be >= 0")
        if work_small_max >= work_large_min:
            raise ValueError("work_small_max must be < work_large_min")

        work = shell_pairs.sp_npair[task_ab].astype(cp.int64) * shell_pairs.sp_npair[task_cd].astype(cp.int64)
        small_mask = work <= int(work_small_max)
        large_mask = work >= int(work_large_min)
        med_mask = ~(small_mask | large_mask)

        small_idx = cp.nonzero(small_mask)[0]
        med_idx = cp.nonzero(med_mask)[0]
        large_idx = cp.nonzero(large_mask)[0]

        if int(small_idx.size) > 0:
            ab_small = cp.ascontiguousarray(task_ab[small_idx])
            cd_small = cp.ascontiguousarray(task_cd[small_idx])
            out_small = cp.empty((int(ab_small.shape[0]), 27), dtype=cp.float64)
            _ext.eri_ppps_warp_inplace_device(
                ab_small,
                cd_small,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out_small.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            out[small_idx] = out_small

        if int(med_idx.size) > 0:
            ab_med = cp.ascontiguousarray(task_ab[med_idx])
            cd_med = cp.ascontiguousarray(task_cd[med_idx])
            out_med = cp.empty((int(ab_med.shape[0]), 27), dtype=cp.float64)
            _ext.eri_ppps_inplace_device(
                ab_med,
                cd_med,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out_med.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            out[med_idx] = out_med

        if int(large_idx.size) > 0:
            if blocks_per_task <= 0:
                raise ValueError("blocks_per_task must be > 0")
            ab_large = cp.ascontiguousarray(task_ab[large_idx])
            cd_large = cp.ascontiguousarray(task_cd[large_idx])
            out_large = cp.empty((int(ab_large.shape[0]), 27), dtype=cp.float64)
            partial = cp.empty((int(ab_large.shape[0]) * int(blocks_per_task) * 27,), dtype=cp.float64)
            _ext.eri_ppps_multiblock_inplace_device(
                ab_large,
                cd_large,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                partial,
                int(blocks_per_task),
                out_large.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            out[large_idx] = out_large
        return out.ravel()


def eri_pppp_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    """Evaluate contracted (pp|pp) shell-quartet tiles (Cartesian) for tasks of class (1,1,1,1).

    Output layout: a flat device array of length `ntasks*81`, with `AB=9` and `CD=9` stored row-major
    as `tile[ab, cd]` where `ab = ia*3 + ib` (A-major, B-minor), `cd = ic*3 + id` (C-major, D-minor).
    """

    import cupy as cp

    _require_cuda_ext()
    with stream_ctx(stream):
        task_ab = cp.asarray(tasks.task_spAB, dtype=cp.int32)
        task_cd = cp.asarray(tasks.task_spCD, dtype=cp.int32)
        task_ab = cp.ascontiguousarray(task_ab)
        task_cd = cp.ascontiguousarray(task_cd)

        out = cp.empty((tasks.ntasks, 81), dtype=cp.float64)
        if tasks.ntasks == 0:
            return out.ravel()

        mode = mode.lower().strip()
        if mode not in ("block", "warp", "multiblock", "auto"):
            raise ValueError("mode must be one of: 'block', 'warp', 'multiblock', 'auto'")

        if mode == "block":
            _ext.eri_pppp_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        if mode == "warp":
            _ext.eri_pppp_warp_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        if mode == "multiblock":
            if blocks_per_task <= 0:
                raise ValueError("blocks_per_task must be > 0")
            partial = cp.empty((tasks.ntasks * int(blocks_per_task) * 81,), dtype=cp.float64)
            _ext.eri_pppp_multiblock_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                partial,
                int(blocks_per_task),
                out.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        # mode == "auto": bin by a crude work proxy.
        if work_small_max < 0 or work_large_min < 0:
            raise ValueError("work_small_max/work_large_min must be >= 0")
        if work_small_max >= work_large_min:
            raise ValueError("work_small_max must be < work_large_min")

        work = shell_pairs.sp_npair[task_ab].astype(cp.int64) * shell_pairs.sp_npair[task_cd].astype(cp.int64)
        small_mask = work <= int(work_small_max)
        large_mask = work >= int(work_large_min)
        med_mask = ~(small_mask | large_mask)

        small_idx = cp.nonzero(small_mask)[0]
        med_idx = cp.nonzero(med_mask)[0]
        large_idx = cp.nonzero(large_mask)[0]

        if int(small_idx.size) > 0:
            ab_small = cp.ascontiguousarray(task_ab[small_idx])
            cd_small = cp.ascontiguousarray(task_cd[small_idx])
            out_small = cp.empty((int(ab_small.shape[0]), 81), dtype=cp.float64)
            _ext.eri_pppp_warp_inplace_device(
                ab_small,
                cd_small,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out_small.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            out[small_idx] = out_small

        if int(med_idx.size) > 0:
            ab_med = cp.ascontiguousarray(task_ab[med_idx])
            cd_med = cp.ascontiguousarray(task_cd[med_idx])
            out_med = cp.empty((int(ab_med.shape[0]), 81), dtype=cp.float64)
            _ext.eri_pppp_inplace_device(
                ab_med,
                cd_med,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out_med.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            out[med_idx] = out_med

        if int(large_idx.size) > 0:
            if blocks_per_task <= 0:
                raise ValueError("blocks_per_task must be > 0")
            ab_large = cp.ascontiguousarray(task_ab[large_idx])
            cd_large = cp.ascontiguousarray(task_cd[large_idx])
            out_large = cp.empty((int(ab_large.shape[0]), 81), dtype=cp.float64)
            partial = cp.empty((int(ab_large.shape[0]) * int(blocks_per_task) * 81,), dtype=cp.float64)
            _ext.eri_pppp_multiblock_inplace_device(
                ab_large,
                cd_large,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                partial,
                int(blocks_per_task),
                out_large.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            out[large_idx] = out_large
        return out.ravel()


def eri_dsss_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    """Evaluate contracted (ds|ss) shell-quartet tiles (Cartesian) for tasks of class (2,0,0,0).

    Output layout: a flat device array of length `ntasks*6`, with `[xx,xy,xz,yy,yz,zz]` for each task.
    """

    import cupy as cp

    _require_cuda_ext()
    with stream_ctx(stream):
        task_ab = cp.asarray(tasks.task_spAB, dtype=cp.int32)
        task_cd = cp.asarray(tasks.task_spCD, dtype=cp.int32)
        task_ab = cp.ascontiguousarray(task_ab)
        task_cd = cp.ascontiguousarray(task_cd)

        out = cp.empty((tasks.ntasks, 6), dtype=cp.float64)
        if tasks.ntasks == 0:
            return out.ravel()

        mode = mode.lower().strip()
        if mode not in ("block", "warp", "multiblock", "auto"):
            raise ValueError("mode must be one of: 'block', 'warp', 'multiblock', 'auto'")

        if mode == "block":
            _ext.eri_dsss_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        if mode == "warp":
            _ext.eri_dsss_warp_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        if mode == "multiblock":
            if blocks_per_task <= 0:
                raise ValueError("blocks_per_task must be > 0")
            partial = cp.empty((tasks.ntasks * int(blocks_per_task) * 6,), dtype=cp.float64)
            _ext.eri_dsss_multiblock_inplace_device(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                partial,
                int(blocks_per_task),
                out.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        # mode == "auto": bin by a crude work proxy.
        if work_small_max < 0 or work_large_min < 0:
            raise ValueError("work_small_max/work_large_min must be >= 0")
        if work_small_max >= work_large_min:
            raise ValueError("work_small_max must be < work_large_min")

        work = shell_pairs.sp_npair[task_ab].astype(cp.int64) * shell_pairs.sp_npair[task_cd].astype(cp.int64)
        small_mask = work <= int(work_small_max)
        large_mask = work >= int(work_large_min)
        med_mask = ~(small_mask | large_mask)

        small_idx = cp.nonzero(small_mask)[0]
        med_idx = cp.nonzero(med_mask)[0]
        large_idx = cp.nonzero(large_mask)[0]

        if int(small_idx.size) > 0:
            ab_small = cp.ascontiguousarray(task_ab[small_idx])
            cd_small = cp.ascontiguousarray(task_cd[small_idx])
            out_small = cp.empty((int(ab_small.shape[0]), 6), dtype=cp.float64)
            _ext.eri_dsss_warp_inplace_device(
                ab_small,
                cd_small,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out_small.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            out[small_idx] = out_small

        if int(med_idx.size) > 0:
            ab_med = cp.ascontiguousarray(task_ab[med_idx])
            cd_med = cp.ascontiguousarray(task_cd[med_idx])
            out_med = cp.empty((int(ab_med.shape[0]), 6), dtype=cp.float64)
            _ext.eri_dsss_inplace_device(
                ab_med,
                cd_med,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out_med.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            out[med_idx] = out_med

        if int(large_idx.size) > 0:
            if blocks_per_task <= 0:
                raise ValueError("blocks_per_task must be > 0")
            ab_large = cp.ascontiguousarray(task_ab[large_idx])
            cd_large = cp.ascontiguousarray(task_cd[large_idx])
            out_large = cp.empty((int(ab_large.shape[0]), 6), dtype=cp.float64)
            partial = cp.empty((int(ab_large.shape[0]) * int(blocks_per_task) * 6,), dtype=cp.float64)
            _ext.eri_dsss_multiblock_inplace_device(
                ab_large,
                cd_large,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                partial,
                int(blocks_per_task),
                out_large.ravel(),
                int(threads),
                int(_stream_ptr(stream)),
                False,
            )
            out[large_idx] = out_large
        return out.ravel()


def eri_ddss_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=2,
        lb=2,
        lc=0,
        ld=0,
        ext_block_name="eri_ddss_inplace_device",
        ext_warp_name="eri_ddss_warp_inplace_device",
        ext_multiblock_name="eri_ddss_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
        tuned_threads=96,
        allow_generic_fallback=False,
        fallback_invalid_warp_to_block=False,
    )


def _eri_fixed_class_specialized_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    la: int,
    lb: int,
    lc: int,
    ld: int,
    ext_block_name: str,
    ext_warp_name: str,
    ext_multiblock_name: str,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
    tuned_threads: int | None = None,
    allow_generic_fallback: bool = True,
    fallback_invalid_warp_to_block: bool = True,
):
    """Common execution path for fixed-(la,lb,lc,ld) Step-2 kernel classes."""

    import cupy as cp

    _require_cuda_ext()

    la = int(la)
    lb = int(lb)
    lc = int(lc)
    ld = int(ld)
    ncomp = int(ncart(la)) * int(ncart(lb)) * int(ncart(lc)) * int(ncart(ld))
    threads = int(threads)
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if tuned_threads is None:
        # Fixed-class kernels default to warp-aligned sizes near ncomp.
        # Keep caller intent as an upper bound.
        threads_launch = int(min(max(32, ((ncomp + 31) // 32) * 32), min(1024, threads)))
    else:
        threads_launch = int(min(1024, max(32, int(tuned_threads))))
    threads_launch = int(max(32, (threads_launch // 32) * 32))
    warp_supported = ncomp <= 128 and int(threads_launch) <= 256 and (int(threads_launch) % 32) == 0

    mode = mode.lower().strip()
    if mode not in ("block", "warp", "multiblock", "auto"):
        raise ValueError("mode must be one of: 'block', 'warp', 'multiblock', 'auto'")

    has_native = all(hasattr(_ext, name) for name in (ext_block_name, ext_warp_name, ext_multiblock_name))
    if not has_native:
        if not bool(allow_generic_fallback):
            raise RuntimeError(
                "missing required specialized quartet kernels in CUDA extension; "
                "rebuild cuERI extension with generated wave kernels"
            )
        # Compatibility fallback when Python code is newer than the compiled extension.
        if mode == "warp":
            if warp_supported:
                return eri_rys_generic_warp_device(
                    tasks,
                    basis,
                    shell_pairs,
                    pair_tables,
                    la=la,
                    lb=lb,
                    lc=lc,
                    ld=ld,
                    stream=stream,
                    threads=int(threads_launch),
                )
            return eri_rys_generic_device(
                tasks,
                basis,
                shell_pairs,
                pair_tables,
                la=la,
                lb=lb,
                lc=lc,
                ld=ld,
                stream=stream,
                threads=int(threads_launch),
            )
        if mode in ("block", "multiblock"):
            return eri_rys_generic_device(
                tasks,
                basis,
                shell_pairs,
                pair_tables,
                la=la,
                lb=lb,
                lc=lc,
                ld=ld,
                stream=stream,
                threads=int(threads_launch),
            )

        # mode == "auto"
        if warp_supported:
            return eri_rys_generic_warp_device(
                tasks,
                basis,
                shell_pairs,
                pair_tables,
                la=la,
                lb=lb,
                lc=lc,
                ld=ld,
                stream=stream,
                threads=int(threads_launch),
            )
        return eri_rys_generic_device(
            tasks,
            basis,
            shell_pairs,
            pair_tables,
            la=la,
            lb=lb,
            lc=lc,
            ld=ld,
            stream=stream,
            threads=int(threads_launch),
        )

    block_fn = getattr(_ext, ext_block_name)
    warp_fn = getattr(_ext, ext_warp_name)
    multiblock_fn = getattr(_ext, ext_multiblock_name)

    with stream_ctx(stream):
        task_ab = cp.asarray(tasks.task_spAB, dtype=cp.int32)
        task_cd = cp.asarray(tasks.task_spCD, dtype=cp.int32)
        task_ab = cp.ascontiguousarray(task_ab)
        task_cd = cp.ascontiguousarray(task_cd)

        out = cp.empty((tasks.ntasks, ncomp), dtype=cp.float64)
        if tasks.ntasks == 0:
            return out.ravel()

        def _launch_block(ab_idx, cd_idx, out_buf):
            block_fn(
                ab_idx,
                cd_idx,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                out_buf,
                int(threads_launch),
                int(_stream_ptr(stream)),
                False,
            )

        def _launch_warp_or_block(ab_idx, cd_idx, out_buf):
            if not warp_supported:
                _launch_block(ab_idx, cd_idx, out_buf)
                return
            if not bool(fallback_invalid_warp_to_block):
                warp_fn(
                    ab_idx,
                    cd_idx,
                    shell_pairs.sp_A,
                    shell_pairs.sp_B,
                    shell_pairs.sp_pair_start,
                    shell_pairs.sp_npair,
                    basis.shell_cx,
                    basis.shell_cy,
                    basis.shell_cz,
                    pair_tables.pair_eta,
                    pair_tables.pair_Px,
                    pair_tables.pair_Py,
                    pair_tables.pair_Pz,
                    pair_tables.pair_cK,
                    out_buf,
                    int(threads_launch),
                    int(_stream_ptr(stream)),
                    False,
                )
                return
            try:
                warp_fn(
                    ab_idx,
                    cd_idx,
                    shell_pairs.sp_A,
                    shell_pairs.sp_B,
                    shell_pairs.sp_pair_start,
                    shell_pairs.sp_npair,
                    basis.shell_cx,
                    basis.shell_cy,
                    basis.shell_cz,
                    pair_tables.pair_eta,
                    pair_tables.pair_Px,
                    pair_tables.pair_Py,
                    pair_tables.pair_Pz,
                    pair_tables.pair_cK,
                    out_buf,
                    int(threads_launch),
                    int(_stream_ptr(stream)),
                    False,
                )
            except Exception as exc:
                msg = str(exc).lower()
                if "invalid argument" not in msg:
                    raise
                _launch_block(ab_idx, cd_idx, out_buf)

        if mode == "block":
            _launch_block(task_ab, task_cd, out.ravel())
            return out.ravel()

        if mode == "warp":
            _launch_warp_or_block(task_ab, task_cd, out.ravel())
            return out.ravel()

        if mode == "multiblock":
            if blocks_per_task <= 0:
                raise ValueError("blocks_per_task must be > 0")
            partial = cp.empty((tasks.ntasks * int(blocks_per_task) * ncomp,), dtype=cp.float64)
            multiblock_fn(
                task_ab,
                task_cd,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                partial,
                int(blocks_per_task),
                out.ravel(),
                int(threads_launch),
                int(_stream_ptr(stream)),
                False,
            )
            return out.ravel()

        # mode == "auto": bin by a crude work proxy.
        if work_small_max < 0 or work_large_min < 0:
            raise ValueError("work_small_max/work_large_min must be >= 0")
        if work_small_max >= work_large_min:
            raise ValueError("work_small_max must be < work_large_min")

        work = shell_pairs.sp_npair[task_ab].astype(cp.int64) * shell_pairs.sp_npair[task_cd].astype(cp.int64)
        small_mask = work <= int(work_small_max)
        large_mask = work >= int(work_large_min)
        med_mask = ~(small_mask | large_mask)

        small_idx = cp.nonzero(small_mask)[0]
        med_idx = cp.nonzero(med_mask)[0]
        large_idx = cp.nonzero(large_mask)[0]

        if int(small_idx.size) > 0:
            ab_small = cp.ascontiguousarray(task_ab[small_idx])
            cd_small = cp.ascontiguousarray(task_cd[small_idx])
            out_small = cp.empty((int(ab_small.shape[0]), ncomp), dtype=cp.float64)
            _launch_warp_or_block(ab_small, cd_small, out_small.ravel())
            out[small_idx] = out_small

        if int(med_idx.size) > 0:
            ab_med = cp.ascontiguousarray(task_ab[med_idx])
            cd_med = cp.ascontiguousarray(task_cd[med_idx])
            out_med = cp.empty((int(ab_med.shape[0]), ncomp), dtype=cp.float64)
            _launch_block(ab_med, cd_med, out_med.ravel())
            out[med_idx] = out_med

        if int(large_idx.size) > 0:
            if blocks_per_task <= 0:
                raise ValueError("blocks_per_task must be > 0")
            ab_large = cp.ascontiguousarray(task_ab[large_idx])
            cd_large = cp.ascontiguousarray(task_cd[large_idx])
            out_large = cp.empty((int(ab_large.shape[0]), ncomp), dtype=cp.float64)
            partial = cp.empty((int(ab_large.shape[0]) * int(blocks_per_task) * ncomp,), dtype=cp.float64)
            multiblock_fn(
                ab_large,
                cd_large,
                shell_pairs.sp_A,
                shell_pairs.sp_B,
                shell_pairs.sp_pair_start,
                shell_pairs.sp_npair,
                basis.shell_cx,
                basis.shell_cy,
                basis.shell_cz,
                pair_tables.pair_eta,
                pair_tables.pair_Px,
                pair_tables.pair_Py,
                pair_tables.pair_Pz,
                pair_tables.pair_cK,
                partial,
                int(blocks_per_task),
                out_large.ravel(),
                int(threads_launch),
                int(_stream_ptr(stream)),
                False,
            )
            out[large_idx] = out_large
        return out.ravel()


def eri_ssdp_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=0,
        lb=0,
        lc=2,
        ld=1,
        ext_block_name="eri_ssdp_inplace_device",
        ext_warp_name="eri_ssdp_warp_inplace_device",
        ext_multiblock_name="eri_ssdp_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
        tuned_threads=256,
        allow_generic_fallback=False,
        fallback_invalid_warp_to_block=False,
    )


def eri_psds_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=1,
        lb=0,
        lc=2,
        ld=0,
        ext_block_name="eri_psds_inplace_device",
        ext_warp_name="eri_psds_warp_inplace_device",
        ext_multiblock_name="eri_psds_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_psdp_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=1,
        lb=0,
        lc=2,
        ld=1,
        ext_block_name="eri_psdp_inplace_device",
        ext_warp_name="eri_psdp_warp_inplace_device",
        ext_multiblock_name="eri_psdp_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
        tuned_threads=128,
        allow_generic_fallback=False,
        fallback_invalid_warp_to_block=False,
    )


def eri_psdd_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=1,
        lb=0,
        lc=2,
        ld=2,
        ext_block_name="eri_psdd_inplace_device",
        ext_warp_name="eri_psdd_warp_inplace_device",
        ext_multiblock_name="eri_psdd_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
        tuned_threads=160,
        allow_generic_fallback=False,
        fallback_invalid_warp_to_block=False,
    )


def eri_ppds_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=1,
        lb=1,
        lc=2,
        ld=0,
        ext_block_name="eri_ppds_inplace_device",
        ext_warp_name="eri_ppds_warp_inplace_device",
        ext_multiblock_name="eri_ppds_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_ppdp_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=1,
        lb=1,
        lc=2,
        ld=1,
        ext_block_name="eri_ppdp_inplace_device",
        ext_warp_name="eri_ppdp_warp_inplace_device",
        ext_multiblock_name="eri_ppdp_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
        tuned_threads=192,
        allow_generic_fallback=False,
        fallback_invalid_warp_to_block=False,
    )


def eri_ppdd_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=1,
        lb=1,
        lc=2,
        ld=2,
        ext_block_name="eri_ppdd_inplace_device",
        ext_warp_name="eri_ppdd_warp_inplace_device",
        ext_multiblock_name="eri_ppdd_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
        tuned_threads=192,
        allow_generic_fallback=False,
        fallback_invalid_warp_to_block=False,
    )


def eri_dsds_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=2,
        lb=0,
        lc=2,
        ld=0,
        ext_block_name="eri_dsds_inplace_device",
        ext_warp_name="eri_dsds_warp_inplace_device",
        ext_multiblock_name="eri_dsds_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_dsdp_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=2,
        lb=0,
        lc=2,
        ld=1,
        ext_block_name="eri_dsdp_inplace_device",
        ext_warp_name="eri_dsdp_warp_inplace_device",
        ext_multiblock_name="eri_dsdp_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_dsdd_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=2,
        lb=0,
        lc=2,
        ld=2,
        ext_block_name="eri_dsdd_inplace_device",
        ext_warp_name="eri_dsdd_warp_inplace_device",
        ext_multiblock_name="eri_dsdd_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_fpss_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=3,
        lb=1,
        lc=0,
        ld=0,
        ext_block_name="eri_fpss_inplace_device",
        ext_warp_name="eri_fpss_warp_inplace_device",
        ext_multiblock_name="eri_fpss_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_fdss_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=3,
        lb=2,
        lc=0,
        ld=0,
        ext_block_name="eri_fdss_inplace_device",
        ext_warp_name="eri_fdss_warp_inplace_device",
        ext_multiblock_name="eri_fdss_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_ffss_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=3,
        lb=3,
        lc=0,
        ld=0,
        ext_block_name="eri_ffss_inplace_device",
        ext_warp_name="eri_ffss_warp_inplace_device",
        ext_multiblock_name="eri_ffss_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_fpps_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=3,
        lb=1,
        lc=1,
        ld=0,
        ext_block_name="eri_fpps_inplace_device",
        ext_warp_name="eri_fpps_warp_inplace_device",
        ext_multiblock_name="eri_fpps_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_fdps_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=3,
        lb=2,
        lc=1,
        ld=0,
        ext_block_name="eri_fdps_inplace_device",
        ext_warp_name="eri_fdps_warp_inplace_device",
        ext_multiblock_name="eri_fdps_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_ffps_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=3,
        lb=3,
        lc=1,
        ld=0,
        ext_block_name="eri_ffps_inplace_device",
        ext_warp_name="eri_ffps_warp_inplace_device",
        ext_multiblock_name="eri_ffps_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_fpds_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=3,
        lb=1,
        lc=2,
        ld=0,
        ext_block_name="eri_fpds_inplace_device",
        ext_warp_name="eri_fpds_warp_inplace_device",
        ext_multiblock_name="eri_fpds_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_fdds_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=3,
        lb=2,
        lc=2,
        ld=0,
        ext_block_name="eri_fdds_inplace_device",
        ext_warp_name="eri_fdds_warp_inplace_device",
        ext_multiblock_name="eri_fdds_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_ffds_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=3,
        lb=3,
        lc=2,
        ld=0,
        ext_block_name="eri_ffds_inplace_device",
        ext_warp_name="eri_ffds_warp_inplace_device",
        ext_multiblock_name="eri_ffds_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_ssfs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=0,
        lb=0,
        lc=3,
        ld=0,
        ext_block_name="eri_ssfs_inplace_device",
        ext_warp_name="eri_ssfs_warp_inplace_device",
        ext_multiblock_name="eri_ssfs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_psfs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=1,
        lb=0,
        lc=3,
        ld=0,
        ext_block_name="eri_psfs_inplace_device",
        ext_warp_name="eri_psfs_warp_inplace_device",
        ext_multiblock_name="eri_psfs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_ppfs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=1,
        lb=1,
        lc=3,
        ld=0,
        ext_block_name="eri_ppfs_inplace_device",
        ext_warp_name="eri_ppfs_warp_inplace_device",
        ext_multiblock_name="eri_ppfs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_dsfs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=2,
        lb=0,
        lc=3,
        ld=0,
        ext_block_name="eri_dsfs_inplace_device",
        ext_warp_name="eri_dsfs_warp_inplace_device",
        ext_multiblock_name="eri_dsfs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_fsfs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=3,
        lb=0,
        lc=3,
        ld=0,
        ext_block_name="eri_fsfs_inplace_device",
        ext_warp_name="eri_fsfs_warp_inplace_device",
        ext_multiblock_name="eri_fsfs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_dpfs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=2,
        lb=1,
        lc=3,
        ld=0,
        ext_block_name="eri_dpfs_inplace_device",
        ext_warp_name="eri_dpfs_warp_inplace_device",
        ext_multiblock_name="eri_dpfs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_fpfs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=3,
        lb=1,
        lc=3,
        ld=0,
        ext_block_name="eri_fpfs_inplace_device",
        ext_warp_name="eri_fpfs_warp_inplace_device",
        ext_multiblock_name="eri_fpfs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_ddfs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=2,
        lb=2,
        lc=3,
        ld=0,
        ext_block_name="eri_ddfs_inplace_device",
        ext_warp_name="eri_ddfs_warp_inplace_device",
        ext_multiblock_name="eri_ddfs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_fdfs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=3,
        lb=2,
        lc=3,
        ld=0,
        ext_block_name="eri_fdfs_inplace_device",
        ext_warp_name="eri_fdfs_warp_inplace_device",
        ext_multiblock_name="eri_fdfs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_fffs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=3,
        lb=3,
        lc=3,
        ld=0,
        ext_block_name="eri_fffs_inplace_device",
        ext_warp_name="eri_fffs_warp_inplace_device",
        ext_multiblock_name="eri_fffs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_ssgs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=0,
        lb=0,
        lc=4,
        ld=0,
        ext_block_name="eri_ssgs_inplace_device",
        ext_warp_name="eri_ssgs_warp_inplace_device",
        ext_multiblock_name="eri_ssgs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_psgs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=1,
        lb=0,
        lc=4,
        ld=0,
        ext_block_name="eri_psgs_inplace_device",
        ext_warp_name="eri_psgs_warp_inplace_device",
        ext_multiblock_name="eri_psgs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_ppgs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=1,
        lb=1,
        lc=4,
        ld=0,
        ext_block_name="eri_ppgs_inplace_device",
        ext_warp_name="eri_ppgs_warp_inplace_device",
        ext_multiblock_name="eri_ppgs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_dsgs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=2,
        lb=0,
        lc=4,
        ld=0,
        ext_block_name="eri_dsgs_inplace_device",
        ext_warp_name="eri_dsgs_warp_inplace_device",
        ext_multiblock_name="eri_dsgs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_fsgs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=3,
        lb=0,
        lc=4,
        ld=0,
        ext_block_name="eri_fsgs_inplace_device",
        ext_warp_name="eri_fsgs_warp_inplace_device",
        ext_multiblock_name="eri_fsgs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_dpgs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=2,
        lb=1,
        lc=4,
        ld=0,
        ext_block_name="eri_dpgs_inplace_device",
        ext_warp_name="eri_dpgs_warp_inplace_device",
        ext_multiblock_name="eri_dpgs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_fpgs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=3,
        lb=1,
        lc=4,
        ld=0,
        ext_block_name="eri_fpgs_inplace_device",
        ext_warp_name="eri_fpgs_warp_inplace_device",
        ext_multiblock_name="eri_fpgs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_ddgs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=2,
        lb=2,
        lc=4,
        ld=0,
        ext_block_name="eri_ddgs_inplace_device",
        ext_warp_name="eri_ddgs_warp_inplace_device",
        ext_multiblock_name="eri_ddgs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_fdgs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=3,
        lb=2,
        lc=4,
        ld=0,
        ext_block_name="eri_fdgs_inplace_device",
        ext_warp_name="eri_fdgs_warp_inplace_device",
        ext_multiblock_name="eri_fdgs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_ffgs_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=3,
        lb=3,
        lc=4,
        ld=0,
        ext_block_name="eri_ffgs_inplace_device",
        ext_warp_name="eri_ffgs_warp_inplace_device",
        ext_multiblock_name="eri_ffgs_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
    )


def eri_dpdp_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=2,
        lb=1,
        lc=2,
        ld=1,
        ext_block_name="eri_dpdp_inplace_device",
        ext_warp_name="eri_dpdp_warp_inplace_device",
        ext_multiblock_name="eri_dpdp_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
        tuned_threads=192,
        allow_generic_fallback=False,
        fallback_invalid_warp_to_block=False,
    )


def eri_dpdd_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=2,
        lb=1,
        lc=2,
        ld=2,
        ext_block_name="eri_dpdd_inplace_device",
        ext_warp_name="eri_dpdd_warp_inplace_device",
        ext_multiblock_name="eri_dpdd_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
        tuned_threads=128,
        allow_generic_fallback=False,
        fallback_invalid_warp_to_block=False,
    )


def eri_dddd_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    stream=None,
    threads: int = 256,
    mode: str = "block",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
):
    return _eri_fixed_class_specialized_device(
        tasks,
        basis,
        shell_pairs,
        pair_tables,
        la=2,
        lb=2,
        lc=2,
        ld=2,
        ext_block_name="eri_dddd_inplace_device",
        ext_warp_name="eri_dddd_warp_inplace_device",
        ext_multiblock_name="eri_dddd_multiblock_inplace_device",
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
        tuned_threads=256,
        allow_generic_fallback=False,
        fallback_invalid_warp_to_block=False,
    )


def eri_rys_generic_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    la: int,
    lb: int,
    lc: int,
    ld: int,
    stream=None,
    threads: int = 256,
):
    """Evaluate contracted Cartesian shell-quartet tiles using the generic Rys microkernel.

    Notes
    - This is a reference-oriented generic implementation (not yet optimized).
    - Current CUDA implementation supports l<=CUDA_MAX_L on each shell and nroots<=CUDA_MAX_NROOTS.
    - Output layout: flat device array of length `ntasks * nAB * nCD`, where:
      - nAB = ncart(la) * ncart(lb)
      - nCD = ncart(lc) * ncart(ld)
    """

    import cupy as cp

    _require_cuda_ext()

    la = int(la)
    lb = int(lb)
    lc = int(lc)
    ld = int(ld)
    if la < 0 or lb < 0 or lc < 0 or ld < 0:
        raise ValueError("la/lb/lc/ld must be >= 0")
    if max(la, lb, lc, ld) > CUDA_MAX_L:
        raise NotImplementedError(f"eri_rys_generic_device currently supports only l<={CUDA_MAX_L} per shell")

    nroots = ((la + lb + lc + ld) // 2) + 1
    if nroots < 1 or nroots > CUDA_MAX_NROOTS:
        raise NotImplementedError(
            f"eri_rys_generic_device requires nroots in [1,{CUDA_MAX_NROOTS}] "
            f"(lsum<={2 * (CUDA_MAX_NROOTS - 1)})"
        )

    with stream_ctx(stream):
        task_ab = cp.asarray(tasks.task_spAB, dtype=cp.int32)
        task_cd = cp.asarray(tasks.task_spCD, dtype=cp.int32)
        task_ab = cp.ascontiguousarray(task_ab)
        task_cd = cp.ascontiguousarray(task_cd)

        nAB = int(ncart(la)) * int(ncart(lb))
        nCD = int(ncart(lc)) * int(ncart(ld))
        out = cp.empty((tasks.ntasks, nAB * nCD), dtype=cp.float64)
        if tasks.ntasks == 0:
            return out.ravel()

        threads = int(threads)
        if threads <= 0:
            raise ValueError("threads must be > 0")
        # Reduce barrier/idle overhead for tiny tiles.
        n_elem = int(nAB * nCD)
        if threads > n_elem and n_elem > 0:
            want = ((n_elem + 31) // 32) * 32  # next multiple of warp size
            threads = min(threads, max(32, want))

        _ext.eri_rys_generic_inplace_device(
            task_ab,
            task_cd,
            shell_pairs.sp_A,
            shell_pairs.sp_B,
            shell_pairs.sp_pair_start,
            shell_pairs.sp_npair,
            basis.shell_cx,
            basis.shell_cy,
            basis.shell_cz,
            pair_tables.pair_eta,
            pair_tables.pair_Px,
            pair_tables.pair_Py,
            pair_tables.pair_Pz,
            pair_tables.pair_cK,
            int(la),
            int(lb),
            int(lc),
            int(ld),
            out.ravel(),
            int(threads),
            int(_stream_ptr(stream)),
            False,
        )
        return out.ravel()


def eri_rys_generic_warp_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    la: int,
    lb: int,
    lc: int,
    ld: int,
    stream=None,
    threads: int = 256,
):
    """Evaluate contracted Cartesian shell-quartet tiles using a warp-per-task generic Rys kernel.

    Current kernel constraints
    - l<=CUDA_MAX_L for each shell
    - nroots<=CUDA_MAX_NROOTS (lsum<=2*(CUDA_MAX_NROOTS-1))
    - nElem = nAB*nCD <= 128
    - threads must be a multiple of 32 and <=256
    """

    import cupy as cp

    _require_cuda_ext()

    la = int(la)
    lb = int(lb)
    lc = int(lc)
    ld = int(ld)
    if la < 0 or lb < 0 or lc < 0 or ld < 0:
        raise ValueError("la/lb/lc/ld must be >= 0")
    if max(la, lb, lc, ld) > CUDA_MAX_L:
        raise NotImplementedError(f"eri_rys_generic_warp_device currently supports only l<={CUDA_MAX_L} per shell")

    nroots = ((la + lb + lc + ld) // 2) + 1
    if nroots < 1 or nroots > CUDA_MAX_NROOTS:
        raise NotImplementedError(
            f"eri_rys_generic_warp_device requires nroots in [1,{CUDA_MAX_NROOTS}] "
            f"(lsum<={2 * (CUDA_MAX_NROOTS - 1)})"
        )

    nAB = int(ncart(la)) * int(ncart(lb))
    nCD = int(ncart(lc)) * int(ncart(ld))
    n_elem = int(nAB * nCD)
    if n_elem > 128:
        raise ValueError("eri_rys_generic_warp_device requires nAB*nCD <= 128")

    threads = int(threads)
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if threads > 256:
        raise ValueError("eri_rys_generic_warp_device requires threads <= 256")
    if (threads % 32) != 0:
        raise ValueError("threads must be a multiple of 32")

    with stream_ctx(stream):
        task_ab = cp.asarray(tasks.task_spAB, dtype=cp.int32)
        task_cd = cp.asarray(tasks.task_spCD, dtype=cp.int32)
        task_ab = cp.ascontiguousarray(task_ab)
        task_cd = cp.ascontiguousarray(task_cd)

        out = cp.empty((tasks.ntasks, nAB * nCD), dtype=cp.float64)
        if tasks.ntasks == 0:
            return out.ravel()

        _ext.eri_rys_generic_warp_inplace_device(
            task_ab,
            task_cd,
            shell_pairs.sp_A,
            shell_pairs.sp_B,
            shell_pairs.sp_pair_start,
            shell_pairs.sp_npair,
            basis.shell_cx,
            basis.shell_cy,
            basis.shell_cz,
            pair_tables.pair_eta,
            pair_tables.pair_Px,
            pair_tables.pair_Py,
            pair_tables.pair_Pz,
            pair_tables.pair_cK,
            int(la),
            int(lb),
            int(lc),
            int(ld),
            out.ravel(),
            int(threads),
            int(_stream_ptr(stream)),
            False,
        )
        return out.ravel()


def eri_rys_generic_deriv_contracted_device(
    task_spAB: cp.ndarray,
    task_spCD: cp.ndarray,
    dsp: DeviceShellPairs,
    dbasis: DeviceBasisSS,
    pt: DevicePairTables,
    la: int,
    lb: int,
    lc: int,
    ld: int,
    bar_eri: cp.ndarray,
    out: cp.ndarray | None = None,
    threads: int = 256,
    stream: cp.cuda.Stream | None = None,
    sync: bool = True,
) -> cp.ndarray:
    """Contract analytic 4c ERI nuclear derivatives against `bar_eri` on GPU (generic Rys).

    Parameters
    ----------
    task_spAB, task_spCD
        int32 arrays of shape (ntasks,) that select shell pairs for each task.
    dsp, dbasis, pt
        GPU-resident shell-pair tables and basis/pair-table SoA.
    la, lb, lc, ld
        Angular momenta for the batch (must match the task class).
    bar_eri
        float64 array of shape (ntasks, nAB, nCD) or (ntasks, nAB*nCD) or flat (ntasks*nAB*nCD,).
        Layout must match the value tile layout (row-major over (nAB,nCD)).
    out
        Optional float64 output array of shape (ntasks,12) or flat (ntasks*12,).
    threads
        CUDA threads per block (must be multiple of 32, <=256).

    Returns
    -------
    out_reshaped : cp.ndarray
        float64 array of shape (ntasks, 4, 3) with center order [A,B,C,D] and xyz.
    """
    import cupy as cp

    _require_cuda_ext()
    if task_spAB.dtype != cp.int32 or task_spCD.dtype != cp.int32:
        raise TypeError("task_spAB/task_spCD must be int32")
    if bar_eri.dtype != cp.float64:
        raise TypeError("bar_eri must be float64")

    ntasks = int(task_spAB.size)
    if int(task_spCD.size) != ntasks:
        raise ValueError("task_spAB and task_spCD must have the same length")
    if ntasks == 0:
        return cp.empty((0, 4, 3), dtype=cp.float64)

    # Reduce barrier/idle overhead for tiny tiles (common for NAC shell classes).
    n_elem = int(ncart(int(la)) * ncart(int(lb)) * ncart(int(lc)) * ncart(int(ld)))
    threads = int(threads)
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if threads > n_elem and n_elem > 0:
        want = ((n_elem + 31) // 32) * 32
        threads = min(threads, max(32, want))

    # Flatten bar_eri to 1D (ntasks*nAB*nCD,) as required by the C++ binding.
    if bar_eri.ndim == 3:
        bar_flat = bar_eri.reshape(ntasks, -1).ravel()
    elif bar_eri.ndim == 2:
        bar_flat = bar_eri.ravel()
    elif bar_eri.ndim == 1:
        bar_flat = bar_eri
    else:
        raise ValueError("bar_eri must have ndim 1,2,or 3")

    if out is None:
        out_flat = cp.empty((ntasks * 12,), dtype=cp.float64)
    else:
        if out.dtype != cp.float64:
            raise TypeError("out must be float64")
        out_flat = out.ravel()
        if out_flat.size != ntasks * 12:
            raise ValueError("out must have size ntasks*12")

    stream_ptr = _stream_ptr(stream)
    _ext.eri_rys_generic_deriv_contracted_inplace_device(
        task_spAB,
        task_spCD,
        dsp.sp_A,
        dsp.sp_B,
        dsp.sp_pair_start,
        dsp.sp_npair,
        int(la),
        int(lb),
        int(lc),
        int(ld),
        dbasis.shell_cx,
        dbasis.shell_cy,
        dbasis.shell_cz,
        dbasis.shell_prim_start,
        dbasis.shell_nprim,
        dbasis.prim_exp,
        pt.pair_eta,
        pt.pair_Px,
        pt.pair_Py,
        pt.pair_Pz,
        pt.pair_cK,
        bar_flat,
        out_flat,
        int(threads),
        stream_ptr,
        sync,
    )
    return out_flat.reshape(ntasks, 4, 3)


def eri_rys_generic_deriv_contracted_atom_grad_inplace_device(
    task_spAB: cp.ndarray,
    task_spCD: cp.ndarray,
    dsp: DeviceShellPairs,
    dbasis: DeviceBasisSS,
    pt: DevicePairTables,
    la: int,
    lb: int,
    lc: int,
    ld: int,
    bar_eri: cp.ndarray,
    shell_atom: cp.ndarray,
    grad_out: cp.ndarray,
    threads: int = 256,
    stream: cp.cuda.Stream | None = None,
    sync: bool = True,
) -> cp.ndarray:
    """Contract analytic 4c ERI derivatives and accumulate directly to atom gradients on GPU."""

    import cupy as cp

    _require_cuda_ext()
    if task_spAB.dtype != cp.int32 or task_spCD.dtype != cp.int32:
        raise TypeError("task_spAB/task_spCD must be int32")
    if bar_eri.dtype != cp.float64:
        raise TypeError("bar_eri must be float64")
    if shell_atom.dtype != cp.int32:
        raise TypeError("shell_atom must be int32")
    if grad_out.dtype != cp.float64:
        raise TypeError("grad_out must be float64")

    ntasks = int(task_spAB.size)
    if int(task_spCD.size) != ntasks:
        raise ValueError("task_spAB and task_spCD must have the same length")
    if ntasks == 0:
        return grad_out

    n_elem = int(ncart(int(la)) * ncart(int(lb)) * ncart(int(lc)) * ncart(int(ld)))
    threads = int(threads)
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if threads > n_elem and n_elem > 0:
        want = ((n_elem + 31) // 32) * 32
        threads = min(threads, max(32, want))

    if bar_eri.ndim == 3:
        bar_flat = bar_eri.reshape(ntasks, -1).ravel()
    elif bar_eri.ndim == 2:
        bar_flat = bar_eri.ravel()
    elif bar_eri.ndim == 1:
        bar_flat = bar_eri
    else:
        raise ValueError("bar_eri must have ndim 1,2,or 3")

    grad_flat = grad_out.ravel()
    if int(grad_flat.size) <= 0 or int(grad_flat.size) % 3 != 0:
        raise ValueError("grad_out must have shape (natm,3) or flat size natm*3")

    _ext.eri_rys_generic_deriv_contracted_atom_grad_inplace_device(
        task_spAB,
        task_spCD,
        dsp.sp_A,
        dsp.sp_B,
        dsp.sp_pair_start,
        dsp.sp_npair,
        int(la),
        int(lb),
        int(lc),
        int(ld),
        dbasis.shell_cx,
        dbasis.shell_cy,
        dbasis.shell_cz,
        dbasis.shell_prim_start,
        dbasis.shell_nprim,
        dbasis.prim_exp,
        pt.pair_eta,
        pt.pair_Px,
        pt.pair_Py,
        pt.pair_Pz,
        pt.pair_cK,
        bar_flat,
        shell_atom.ravel(),
        grad_flat,
        int(threads),
        _stream_ptr(stream),
        sync,
    )
    return grad_out


def eri_rys_df_ld0_warp_device(
    tasks: TaskList,
    basis: DeviceBasisSS,
    shell_pairs: DeviceShellPairs,
    pair_tables: DevicePairTables,
    *,
    la: int,
    lb: int,
    lc: int,
    stream=None,
    threads: int = 256,
):
    """Evaluate contracted Cartesian tiles for (la,lb,lc,0) using a DF-oriented warp-per-task kernel.

    Output layout: flat device array of length `ntasks * nAB * nC`, where:
      - nAB = ncart(la) * ncart(lb)
      - nC  = ncart(lc)

    Current kernel constraints
    - l<=CUDA_MAX_L for each shell
    - nroots<=CUDA_MAX_NROOTS (lsum<=2*(CUDA_MAX_NROOTS-1))
    - nElem = nAB*nC <= 256
    - threads must be a multiple of 32 and <=256
    """

    import cupy as cp

    _require_cuda_ext()

    la = int(la)
    lb = int(lb)
    lc = int(lc)
    if la < 0 or lb < 0 or lc < 0:
        raise ValueError("la/lb/lc must be >= 0")
    if max(la, lb, lc) > CUDA_MAX_L:
        raise NotImplementedError(f"eri_rys_df_ld0_warp_device currently supports only l<={CUDA_MAX_L} per shell")

    nroots = ((la + lb + lc) // 2) + 1
    if nroots < 1 or nroots > CUDA_MAX_NROOTS:
        raise NotImplementedError(
            f"eri_rys_df_ld0_warp_device requires nroots in [1,{CUDA_MAX_NROOTS}] "
            f"(lsum<={2 * (CUDA_MAX_NROOTS - 1)})"
        )

    nAB = int(ncart(la)) * int(ncart(lb))
    nC = int(ncart(lc))
    n_elem = int(nAB * nC)
    if n_elem > 256:
        raise ValueError("eri_rys_df_ld0_warp_device requires nAB*nC <= 256")

    threads = int(threads)
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if threads > 256:
        raise ValueError("eri_rys_df_ld0_warp_device requires threads <= 256")
    if (threads % 32) != 0:
        raise ValueError("threads must be a multiple of 32")

    with stream_ctx(stream):
        task_ab = cp.asarray(tasks.task_spAB, dtype=cp.int32)
        task_cd = cp.asarray(tasks.task_spCD, dtype=cp.int32)
        task_ab = cp.ascontiguousarray(task_ab)
        task_cd = cp.ascontiguousarray(task_cd)

        out = cp.empty((tasks.ntasks, nAB * nC), dtype=cp.float64)
        if tasks.ntasks == 0:
            return out.ravel()

        _ext.eri_rys_df_ld0_warp_inplace_device(
            task_ab,
            task_cd,
            shell_pairs.sp_A,
            shell_pairs.sp_B,
            shell_pairs.sp_pair_start,
            shell_pairs.sp_npair,
            basis.shell_cx,
            basis.shell_cy,
            basis.shell_cz,
            pair_tables.pair_eta,
            pair_tables.pair_Px,
            pair_tables.pair_Py,
            pair_tables.pair_Pz,
            pair_tables.pair_cK,
            int(la),
            int(lb),
            int(lc),
            out.ravel(),
            int(threads),
            int(_stream_ptr(stream)),
            False,
        )
        return out.ravel()


def rys_roots_weights_device(T, nroots: int, *, stream=None, threads: int = 256):
    """Compute Rys roots/weights for nroots in {1..CUDA_MAX_NROOTS} on GPU.

    Returns `(roots, weights)` as 1D device arrays of length `nT*nroots` (row-major).
    """

    import cupy as cp

    _require_cuda_ext()
    if int(nroots) < 1 or int(nroots) > CUDA_MAX_NROOTS:
        raise ValueError(f"nroots must be in [1, {CUDA_MAX_NROOTS}]")

    with stream_ctx(stream):
        T = cp.asarray(T, dtype=cp.float64).ravel()
        T = cp.ascontiguousarray(T)
        nT = int(T.shape[0])

        roots = cp.empty((nT * int(nroots),), dtype=cp.float64)
        weights = cp.empty((nT * int(nroots),), dtype=cp.float64)
        _ext.rys_roots_weights_inplace_device(
            T,
            roots,
            weights,
            int(nroots),
            int(threads),
            int(_stream_ptr(stream)),
            False,
        )
        return roots, weights


def build_entry_csr_device(tasks: TaskList, *, nkey: int, stream=None, threads: int = 256):
    """Build entry CSR for the Step-1 two-entry trick.

    Returns `(entry_offsets, entry_task, entry_widx)`.
    """

    import cupy as cp

    _require_cuda_ext()
    with stream_ctx(stream):
        task_ab = cp.asarray(tasks.task_spAB, dtype=cp.int32)
        task_cd = cp.asarray(tasks.task_spCD, dtype=cp.int32)
        task_ab = cp.ascontiguousarray(task_ab)
        task_cd = cp.ascontiguousarray(task_cd)

        counts = cp.zeros((nkey,), dtype=cp.int32)
        _ext.count_entries_inplace_device(task_ab, task_cd, counts, int(threads), int(_stream_ptr(stream)), False)

        entry_offsets = cp.empty((nkey + 1,), dtype=cp.int32)
        entry_offsets[0] = np.int32(0)
        entry_offsets[1:] = cp.cumsum(counts, dtype=cp.int32)
        n_entry = int(entry_offsets[-1].item())

        entry_task = cp.empty((n_entry,), dtype=cp.int32)
        entry_widx = cp.empty((n_entry,), dtype=cp.int32)
        cursor = cp.zeros((nkey,), dtype=cp.int32)

        _ext.fill_entry_csr_inplace_device(
            task_ab,
            task_cd,
            entry_offsets,
            cursor,
            entry_task,
            entry_widx,
            int(threads),
            int(_stream_ptr(stream)),
            False,
        )
        return entry_offsets, entry_task, entry_widx


def reduce_from_entry_csr_device(
    entry_offsets,
    entry_task,
    entry_widx,
    eri_task,
    W,
    *,
    stream=None,
    threads: int = 256,
):
    import cupy as cp

    _require_cuda_ext()
    with stream_ctx(stream):
        entry_offsets = cp.asarray(entry_offsets, dtype=cp.int32)
        entry_task = cp.asarray(entry_task, dtype=cp.int32)
        entry_widx = cp.asarray(entry_widx, dtype=cp.int32)
        eri_task = cp.asarray(eri_task, dtype=cp.float64)
        W = cp.asarray(W, dtype=cp.float64)
        entry_offsets = cp.ascontiguousarray(entry_offsets)
        entry_task = cp.ascontiguousarray(entry_task)
        entry_widx = cp.ascontiguousarray(entry_widx)
        eri_task = cp.ascontiguousarray(eri_task)
        W = cp.ascontiguousarray(W)

        nkey = int(W.shape[0])
        Out = cp.empty((nkey,), dtype=cp.float64)
        _ext.reduce_from_entry_csr_inplace_device(
            entry_offsets,
            entry_task,
            entry_widx,
            eri_task,
            W,
            Out,
            int(threads),
            int(_stream_ptr(stream)),
            False,
        )
        return Out


__all__ = [
    "DeviceBasisSS",
    "DevicePairTables",
    "DeviceShellPairs",
    "build_entry_csr_device",
    "build_pair_tables_ss_device",
    "df_int3c2e_ss_device",
    "df_int3c2e_sp_device",
    "df_int3c2e_rys_device",
    "df_metric_2c2e_ss_device",
    "df_metric_2c2e_sp_device",
    "df_metric_2c2e_rys_device",
    "cart2sph_eri_tiles_device",
    "eri_dddd_device",
    "eri_dpdd_device",
    "eri_dpdp_device",
    "eri_dsdd_device",
    "eri_dsdp_device",
    "eri_dsds_device",
    "eri_dsfs_device",
    "eri_dsgs_device",
    "eri_ddss_device",
    "eri_ddfs_device",
    "eri_ddgs_device",
    "eri_dpgs_device",
    "eri_dpfs_device",
    "eri_dsss_device",
    "eri_fdds_device",
    "eri_fdgs_device",
    "eri_fdfs_device",
    "eri_fdps_device",
    "eri_fdss_device",
    "eri_ffds_device",
    "eri_fffs_device",
    "eri_ffgs_device",
    "eri_ffps_device",
    "eri_ffss_device",
    "eri_fpds_device",
    "eri_fpfs_device",
    "eri_fpgs_device",
    "eri_fpps_device",
    "eri_fpss_device",
    "eri_fsfs_device",
    "eri_fsgs_device",
    "eri_ppgs_device",
    "eri_psgs_device",
    "eri_ssgs_device",
    "eri_ppfs_device",
    "eri_psfs_device",
    "eri_ssfs_device",
    "eri_ppdd_device",
    "eri_ppdp_device",
    "eri_ppds_device",
    "eri_rys_df_ld0_warp_device",
    "eri_rys_generic_deriv_contracted_atom_grad_inplace_device",
    "eri_rys_generic_device",
    "eri_rys_generic_warp_device",
    "eri_psdd_device",
    "eri_psdp_device",
    "eri_psds_device",
    "eri_pppp_device",
    "eri_ppps_device",
    "eri_ppss_device",
    "eri_psps_device",
    "eri_psss_device",
    "eri_ssdp_device",
    "eri_ssss_device",
    "has_cuda_ext",
    "reduce_from_entry_csr_device",
    "rys_roots_weights_device",
    "scatter_eri_tiles_sph_s4_inplace_device",
    "scatter_eri_tiles_sph_s8_inplace_device",
    "schwarz_ssss_device",
    "sph_coeff_sph_to_cart_device",
    "to_device_basis_ss",
    "to_device_shell_pairs",
]
