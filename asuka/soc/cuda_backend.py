from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from asuka.cuguga.drt import DRT

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None  # type: ignore

try:  # optional CUDA extension
    from asuka import _guga_cuda_ext as _ext
except Exception:  # pragma: no cover
    _ext = None


def _has_soc_cuda_entrypoints() -> bool:
    return (
        (_ext is not None)
        and hasattr(_ext, "make_triplet_factors_workspace")
        and hasattr(_ext, "triplet_apply_contracted_all_m_from_epq_table_inplace_device")
        and hasattr(_ext, "triplet_apply_contracted_all_m_dfs_inplace_device")
    )


def _has_soc_cuda_rho_entrypoints() -> bool:
    return (
        _has_soc_cuda_entrypoints()
        and (_ext is not None)
        and hasattr(_ext, "triplet_build_rho_all_m_from_epq_table_inplace_device")
        and hasattr(_ext, "triplet_build_rho_all_m_dfs_inplace_device")
    )


def _has_soc_cuda_gm_entrypoints() -> bool:
    return (
        _has_soc_cuda_entrypoints()
        and (_ext is not None)
        and hasattr(_ext, "triplet_build_gm_all_m_from_epq_table_inplace_device")
        and hasattr(_ext, "triplet_build_gm_all_m_dfs_inplace_device")
    )


def has_soc_cuda() -> bool:
    """Return True if SOC CUDA triplet-apply entrypoints are available."""

    if (cp is None) or not _has_soc_cuda_entrypoints():
        return False
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0  # type: ignore[attr-defined]
    except Exception:
        return False


@dataclass(frozen=True)
class TripletFactorsHost:
    twos_max: int
    sixj_211: np.ndarray
    t_factor: np.ndarray


def _delta_idx(delta: int) -> int:
    if delta == +1:
        return 0
    if delta == -1:
        return 1
    raise ValueError("delta must be +1 or -1")


@lru_cache(maxsize=8)
def build_triplet_factors_host(twos_max: int) -> TripletFactorsHost:
    """Build host-side factor tables used by SOC CUDA triplet kernels."""

    twos_max = int(twos_max)
    if twos_max < 0:
        raise ValueError("twos_max must be >= 0")

    from asuka.soc.triplet_factors import T_factor  # noqa: PLC0415
    from asuka.soc.wigner import wigner_6j_twos  # noqa: PLC0415

    n = twos_max + 1
    sixj_211 = np.zeros((n, n, n), dtype=np.float32)
    for d in range(n):
        for e in range(n):
            for f in range(n):
                sixj_211[d, e, f] = np.float32(wigner_6j_twos(2, 1, 1, d, e, f))

    t_factor = np.zeros((2, n, n, 2, 2), dtype=np.float32)
    for op_idx, twos_opline in enumerate((1, 2)):
        for skm1 in range(n):
            for skm1_p in range(n):
                for dk in (+1, -1):
                    sk = skm1 + dk
                    if sk < 0 or sk > twos_max:
                        continue
                    dk_idx = _delta_idx(dk)
                    for db in (+1, -1):
                        sk_p = skm1_p + db
                        if sk_p < 0 or sk_p > twos_max:
                            continue
                        db_idx = _delta_idx(db)
                        t_factor[op_idx, skm1, skm1_p, dk_idx, db_idx] = np.float32(
                            T_factor(
                                twos_Skm1=skm1,
                                twos_Sk=sk,
                                twos_Skm1_p=skm1_p,
                                twos_Sk_p=sk_p,
                                twos_opline=twos_opline,
                            )
                        )

    return TripletFactorsHost(twos_max=twos_max, sixj_211=sixj_211, t_factor=t_factor)


@lru_cache(maxsize=8)
def make_triplet_factors_workspace_device(twos_max: int):
    """Create and cache device workspace for SOC triplet factor tables."""

    if _ext is None or not hasattr(_ext, "make_triplet_factors_workspace"):
        raise RuntimeError("SOC triplet factor workspace entrypoint is not available")
    host = build_triplet_factors_host(int(twos_max))
    return _ext.make_triplet_factors_workspace(
        int(host.twos_max),
        np.asarray(host.sixj_211, dtype=np.float32, order="C"),
        np.asarray(host.t_factor, dtype=np.float32, order="C"),
    )


def _twos_max_for_drts(drts: Sequence[DRT]) -> int:
    mx = 0
    for drt in drts:
        arr = np.asarray(drt.node_twos, dtype=np.int32)
        if arr.size:
            mx = max(mx, int(arr.max()))
    return int(mx)


def _prepare_out(cp_mod, out, *, ncsf_bra: int):
    if out is None:
        return cp_mod.zeros((3, ncsf_bra), dtype=cp_mod.float64)
    out = cp_mod.asarray(out, dtype=cp_mod.float64)
    out = cp_mod.ascontiguousarray(out)
    if out.shape != (3, ncsf_bra):
        raise ValueError("output must have shape (3, ncsf_bra)")
    out[...] = 0.0
    return out


def _stream_ptr(cp_mod, stream) -> int:
    if stream is None:
        return int(cp_mod.cuda.get_current_stream().ptr)
    return int(getattr(stream, "ptr", stream))


def _cpu_fallback_result(
    drt_bra: DRT,
    drt_ket: DRT,
    c_ket,
    h_m,
    *,
    cp_mod,
    out_re,
    out_im,
    block_nops_cpu: int,
):
    from asuka.soc.trdm import apply_contracted_triplet_all_m  # noqa: PLC0415

    c_host = np.asarray(cp_mod.asnumpy(cp_mod.asarray(c_ket, dtype=cp_mod.float64)), dtype=np.float64)
    h_host = np.asarray(cp_mod.asnumpy(cp_mod.asarray(h_m, dtype=cp_mod.complex128)), dtype=np.complex128)
    out_host = apply_contracted_triplet_all_m(
        drt_bra,
        drt_ket,
        c_host,
        h_host,
        block_nops=int(block_nops_cpu),
        backend="cpu",
    )
    out_re[...] = cp_mod.asarray(np.asarray(out_host.real, dtype=np.float64), dtype=cp_mod.float64)
    out_im[...] = cp_mod.asarray(np.asarray(out_host.imag, dtype=np.float64), dtype=cp_mod.float64)
    return out_re, out_im


@dataclass
class SOCDeviceContext:
    """Pre-built device structures for SOC CUDA triplet-apply kernels.

    Reuse across multiple kernel calls with the same DRT pair and h_m to avoid
    redundant device transfers and structure setup.
    """

    drt_bra: DRT
    drt_ket: DRT
    drt_bra_dev: object
    drt_ket_dev: object
    ket_state_dev: object
    tf_ws: object
    h_re: object  # (3, norb, norb) device float64, transposed to kernel convention
    h_im: object  # (3, norb, norb) device float64, transposed to kernel convention
    is_same_drt: bool
    epq_indptr: object  # None for cross-DRT
    epq_indices: object  # None for cross-DRT
    epq_pq: object  # None for cross-DRT
    threads: int


def prepare_soc_device_context(
    drt_bra: DRT,
    drt_ket: DRT,
    h_m,
    *,
    threads: int = 128,
    use_epq_table_if_possible: bool = True,
) -> SOCDeviceContext:
    """Build a reusable device context for SOC triplet-apply kernel calls.

    Parameters
    ----------
    h_m
        SOC integrals with shape ``(3, norb, norb)``, complex128.
        Can be a NumPy or CuPy array.
    """

    if cp is None:
        raise RuntimeError("CuPy is required for SOC CUDA backend")
    if not _has_soc_cuda_entrypoints():
        raise RuntimeError("SOC CUDA kernels are unavailable")
    if int(drt_bra.norb) != int(drt_ket.norb):
        raise ValueError("drt_bra and drt_ket must have the same norb")

    cp_mod = cp  # type: ignore[assignment]
    norb = int(drt_ket.norb)

    h_m_d = cp_mod.asarray(h_m, dtype=cp_mod.complex128)
    if h_m_d.shape != (3, norb, norb):
        raise ValueError("h_m must have shape (3, norb, norb)")
    h_m_t = cp_mod.swapaxes(h_m_d, 1, 2)
    h_re = cp_mod.ascontiguousarray(h_m_t.real.astype(cp_mod.float64, copy=False))
    h_im = cp_mod.ascontiguousarray(h_m_t.imag.astype(cp_mod.float64, copy=False))

    from asuka.cuda.cuda_backend import (  # noqa: PLC0415
        build_epq_action_table_combined_device,
        make_device_drt,
        make_device_state_cache,
    )

    is_same_drt = drt_bra is drt_ket
    drt_bra_dev = make_device_drt(drt_bra)
    drt_ket_dev = drt_bra_dev if is_same_drt else make_device_drt(drt_ket)
    ket_state_dev = make_device_state_cache(drt_ket, drt_ket_dev)
    twos_max = _twos_max_for_drts((drt_bra, drt_ket))
    tf_ws = make_triplet_factors_workspace_device(twos_max)

    epq_indptr = None
    epq_indices = None
    epq_pq = None
    if bool(use_epq_table_if_possible) and is_same_drt:
        epq_indptr, _epq_indices, _epq_pq, _epq_data = build_epq_action_table_combined_device(
            drt_ket,
            drt_ket_dev,
            ket_state_dev,
            threads=int(threads),
            sync=True,
            check_overflow=True,
            use_cache=True,
            dtype=np.float64,
            indptr_dtype="int64",
        )
        epq_indices = cp_mod.ascontiguousarray(cp_mod.asarray(_epq_indices, dtype=cp_mod.int32))
        epq_pq = cp_mod.ascontiguousarray(cp_mod.asarray(_epq_pq, dtype=cp_mod.int32))

    return SOCDeviceContext(
        drt_bra=drt_bra,
        drt_ket=drt_ket,
        drt_bra_dev=drt_bra_dev,
        drt_ket_dev=drt_ket_dev,
        ket_state_dev=ket_state_dev,
        tf_ws=tf_ws,
        h_re=h_re,
        h_im=h_im,
        is_same_drt=is_same_drt,
        epq_indptr=epq_indptr,
        epq_indices=epq_indices,
        epq_pq=epq_pq,
        threads=int(threads),
    )


def _apply_contracted_triplet_all_m_cuda_inner(
    ctx: SOCDeviceContext,
    c_ket_d,
    *,
    out_re=None,
    out_im=None,
    stream=None,
    sync: bool = False,
):
    """Low-level triplet apply using a pre-built :class:`SOCDeviceContext`.

    Parameters
    ----------
    c_ket_d
        1-D device array of length ``ncsf_ket`` (float64).
    out_re, out_im
        Optional pre-allocated device arrays of shape ``(3, ncsf_bra)``.
        If provided they are zeroed before use; if *None*, new arrays are allocated.
    sync
        Whether to synchronize after the kernel (default *False* for loop usage).
    """

    cp_mod = cp  # type: ignore[assignment]
    c_ket_d = cp_mod.ascontiguousarray(cp_mod.asarray(c_ket_d, dtype=cp_mod.float64)).ravel()
    ncsf_bra = int(ctx.drt_bra.ncsf)
    if out_re is None:
        out_re = cp_mod.zeros((3, ncsf_bra), dtype=cp_mod.float64)
    else:
        out_re[...] = 0.0
    if out_im is None:
        out_im = cp_mod.zeros((3, ncsf_bra), dtype=cp_mod.float64)
    else:
        out_im[...] = 0.0

    stream_ptr = _stream_ptr(cp_mod, stream)
    threads = ctx.threads

    if ctx.is_same_drt and ctx.epq_indptr is not None:
        _ext.triplet_apply_contracted_all_m_from_epq_table_inplace_device(
            ctx.drt_ket_dev,
            ctx.ket_state_dev,
            ctx.epq_indptr,
            ctx.epq_indices,
            ctx.epq_pq,
            c_ket_d,
            ctx.h_re,
            ctx.h_im,
            ctx.tf_ws,
            out_re,
            out_im,
            int(threads),
            int(stream_ptr),
            bool(sync),
        )
        return out_re, out_im

    _ext.triplet_apply_contracted_all_m_dfs_inplace_device(
        ctx.drt_bra_dev,
        ctx.drt_ket_dev,
        ctx.ket_state_dev,
        c_ket_d,
        ctx.h_re,
        ctx.h_im,
        ctx.tf_ws,
        out_re,
        out_im,
        int(threads),
        int(stream_ptr),
        bool(sync),
        int(ctx.drt_bra.root),
        int(ctx.drt_bra.leaf),
        int(ctx.drt_bra.twos_target),
        int(ctx.drt_ket.twos_target),
    )
    return out_re, out_im


def apply_contracted_triplet_all_m_cuda(
    drt_bra: DRT,
    drt_ket: DRT,
    c_ket,
    h_m,
    *,
    out_re=None,
    out_im=None,
    threads: int = 128,
    stream=None,
    sync: bool = True,
    use_epq_table_if_possible: bool = True,
    fallback_to_cpu: bool = True,
    block_nops_cpu: int = 8,
):
    """CUDA SOC contracted triplet apply, returning `(out_re, out_im)` device arrays."""

    if cp is None:
        raise RuntimeError("CuPy is required for SOC CUDA backend")
    cp_mod = cp  # type: ignore[assignment]

    ncsf_bra = int(drt_bra.ncsf)
    out_re = _prepare_out(cp_mod, out_re, ncsf_bra=ncsf_bra)
    out_im = _prepare_out(cp_mod, out_im, ncsf_bra=ncsf_bra)

    if not _has_soc_cuda_entrypoints():
        if not bool(fallback_to_cpu):
            raise RuntimeError("SOC CUDA kernels are unavailable in asuka._guga_cuda_ext")
        return _cpu_fallback_result(
            drt_bra,
            drt_ket,
            c_ket,
            h_m,
            cp_mod=cp_mod,
            out_re=out_re,
            out_im=out_im,
            block_nops_cpu=int(block_nops_cpu),
        )

    if int(drt_bra.norb) != int(drt_ket.norb):
        raise ValueError("drt_bra and drt_ket must have the same norb")

    c_ket_d = cp_mod.asarray(c_ket, dtype=cp_mod.float64).ravel()
    if int(c_ket_d.size) != int(drt_ket.ncsf):
        raise ValueError("c_ket has wrong length")

    try:
        ctx = prepare_soc_device_context(
            drt_bra,
            drt_ket,
            h_m,
            threads=int(threads),
            use_epq_table_if_possible=bool(use_epq_table_if_possible),
        )
        return _apply_contracted_triplet_all_m_cuda_inner(
            ctx,
            c_ket_d,
            out_re=out_re,
            out_im=out_im,
            stream=stream,
            sync=bool(sync),
        )
    except Exception:
        if not bool(fallback_to_cpu):
            raise
        return _cpu_fallback_result(
            drt_bra,
            drt_ket,
            c_ket,
            h_m,
            cp_mod=cp_mod,
            out_re=out_re,
            out_im=out_im,
            block_nops_cpu=int(block_nops_cpu),
        )


def build_rho_soc_m_block_cuda(
    drt_bra: DRT,
    drt_ket: DRT,
    c_bra,
    c_ket,
    eta_sub,
    *,
    out_re=None,
    out_im=None,
    threads: int = 128,
    stream=None,
    sync: bool = True,
    use_epq_table_if_possible: bool = True,
):
    """CUDA direct rho builder for one (drt_bra, drt_ket) block.

    Parameters
    ----------
    c_bra
        Real array with shape (ncsf_bra, nb).
    c_ket
        Real array with shape (ncsf_ket, nk).
    eta_sub
        Complex array with shape (3, nb, nk).
    """

    if cp is None:
        raise RuntimeError("CuPy is required for SOC CUDA backend")
    if not _has_soc_cuda_rho_entrypoints():
        raise RuntimeError("SOC CUDA rho kernels are unavailable in asuka._guga_cuda_ext")
    if int(drt_bra.norb) != int(drt_ket.norb):
        raise ValueError("drt_bra and drt_ket must have the same norb")

    cp_mod = cp  # type: ignore[assignment]
    norb = int(drt_bra.norb)
    ncsf_bra = int(drt_bra.ncsf)
    ncsf_ket = int(drt_ket.ncsf)

    c_bra_d = cp_mod.ascontiguousarray(cp_mod.asarray(c_bra, dtype=cp_mod.float64))
    c_ket_d = cp_mod.ascontiguousarray(cp_mod.asarray(c_ket, dtype=cp_mod.float64))
    if c_bra_d.ndim != 2 or int(c_bra_d.shape[0]) != ncsf_bra:
        raise ValueError("c_bra must have shape (ncsf_bra, nb)")
    if c_ket_d.ndim != 2 or int(c_ket_d.shape[0]) != ncsf_ket:
        raise ValueError("c_ket must have shape (ncsf_ket, nk)")
    nb = int(c_bra_d.shape[1])
    nk = int(c_ket_d.shape[1])
    if nb <= 0 or nk <= 0:
        raise ValueError("nb and nk must be >= 1")

    eta_d = cp_mod.ascontiguousarray(cp_mod.asarray(eta_sub, dtype=cp_mod.complex128))
    if eta_d.shape != (3, nb, nk):
        raise ValueError("eta_sub must have shape (3, nb, nk)")
    eta_re = cp_mod.ascontiguousarray(eta_d.real.astype(cp_mod.float64, copy=False))
    eta_im = cp_mod.ascontiguousarray(eta_d.imag.astype(cp_mod.float64, copy=False))

    if out_re is None:
        out_re = cp_mod.zeros((3, norb, norb), dtype=cp_mod.float64)
    else:
        out_re = cp_mod.ascontiguousarray(cp_mod.asarray(out_re, dtype=cp_mod.float64))
        if out_re.shape != (3, norb, norb):
            raise ValueError("out_re must have shape (3,norb,norb)")
        out_re[...] = 0.0
    if out_im is None:
        out_im = cp_mod.zeros((3, norb, norb), dtype=cp_mod.float64)
    else:
        out_im = cp_mod.ascontiguousarray(cp_mod.asarray(out_im, dtype=cp_mod.float64))
        if out_im.shape != (3, norb, norb):
            raise ValueError("out_im must have shape (3,norb,norb)")
        out_im[...] = 0.0

    stream_ptr = _stream_ptr(cp_mod, stream)

    from asuka.cuda.cuda_backend import (  # noqa: PLC0415
        build_epq_action_table_combined_device,
        make_device_drt,
        make_device_state_cache,
    )

    drt_bra_dev = make_device_drt(drt_bra)
    drt_ket_dev = drt_bra_dev if (drt_bra is drt_ket) else make_device_drt(drt_ket)
    ket_state_dev = make_device_state_cache(drt_ket, drt_ket_dev)
    twos_max = _twos_max_for_drts((drt_bra, drt_ket))
    tf_ws = make_triplet_factors_workspace_device(twos_max)

    # CUDA rho kernels accumulate by operator indices (p,q) for T_{pq}.
    # SOC rho convention in this module is qp-adjoint:
    #   rho[p_out, q_out] = sum eta * <bra||T_{q_out p_out}||ket>.
    # Therefore convert kernel output to SOC convention by swapping orbital axes.
    def _to_soc_rho_convention(r_re, r_im):
        return (
            cp_mod.ascontiguousarray(cp_mod.swapaxes(r_re, 1, 2)),
            cp_mod.ascontiguousarray(cp_mod.swapaxes(r_im, 1, 2)),
        )

    if bool(use_epq_table_if_possible) and (drt_bra is drt_ket):
        epq_indptr, epq_indices, epq_pq, _epq_data = build_epq_action_table_combined_device(
            drt_ket,
            drt_ket_dev,
            ket_state_dev,
            threads=int(threads),
            sync=True,
            check_overflow=True,
            use_cache=True,
            dtype=np.float64,
            indptr_dtype="int64",
        )
        epq_indices = cp_mod.ascontiguousarray(cp_mod.asarray(epq_indices, dtype=cp_mod.int32))
        epq_pq = cp_mod.ascontiguousarray(cp_mod.asarray(epq_pq, dtype=cp_mod.int32))
        _ext.triplet_build_rho_all_m_from_epq_table_inplace_device(
            drt_ket_dev,
            ket_state_dev,
            epq_indptr,
            epq_indices,
            epq_pq,
            c_bra_d,
            c_ket_d,
            eta_re,
            eta_im,
            tf_ws,
            out_re,
            out_im,
            int(threads),
            int(stream_ptr),
            bool(sync),
        )
        return _to_soc_rho_convention(out_re, out_im)

    _ext.triplet_build_rho_all_m_dfs_inplace_device(
        drt_bra_dev,
        drt_ket_dev,
        ket_state_dev,
        c_bra_d,
        c_ket_d,
        eta_re,
        eta_im,
        tf_ws,
        out_re,
        out_im,
        int(threads),
        int(stream_ptr),
        bool(sync),
        int(drt_bra.root),
        int(drt_bra.leaf),
        int(drt_bra.twos_target),
        int(drt_ket.twos_target),
    )
    return _to_soc_rho_convention(out_re, out_im)


def build_gm_soc_m_block_cuda(
    drt_bra: DRT,
    drt_ket: DRT,
    c_bra,
    c_ket,
    h_m,
    *,
    out_re=None,
    out_im=None,
    threads: int = 128,
    stream=None,
    sync: bool = True,
    use_epq_table_if_possible: bool = True,
):
    """CUDA direct G_m block builder for one (drt_bra, drt_ket) pair.

    Returns
    -------
    out_re, out_im
        Real/imag device arrays with shape (3, nb, nk).
    """

    if cp is None:
        raise RuntimeError("CuPy is required for SOC CUDA backend")
    if not _has_soc_cuda_gm_entrypoints():
        raise RuntimeError("SOC CUDA Gm kernels are unavailable in asuka._guga_cuda_ext")
    if int(drt_bra.norb) != int(drt_ket.norb):
        raise ValueError("drt_bra and drt_ket must have the same norb")

    cp_mod = cp  # type: ignore[assignment]
    norb = int(drt_bra.norb)
    ncsf_bra = int(drt_bra.ncsf)
    ncsf_ket = int(drt_ket.ncsf)

    c_bra_d = cp_mod.ascontiguousarray(cp_mod.asarray(c_bra, dtype=cp_mod.float64))
    c_ket_d = cp_mod.ascontiguousarray(cp_mod.asarray(c_ket, dtype=cp_mod.float64))
    if c_bra_d.ndim != 2 or int(c_bra_d.shape[0]) != ncsf_bra:
        raise ValueError("c_bra must have shape (ncsf_bra, nb)")
    if c_ket_d.ndim != 2 or int(c_ket_d.shape[0]) != ncsf_ket:
        raise ValueError("c_ket must have shape (ncsf_ket, nk)")
    nb = int(c_bra_d.shape[1])
    nk = int(c_ket_d.shape[1])
    if nb <= 0 or nk <= 0:
        raise ValueError("nb and nk must be >= 1")

    h_m_d = cp_mod.asarray(h_m, dtype=cp_mod.complex128)
    if h_m_d.shape != (3, norb, norb):
        raise ValueError("h_m must have shape (3,norb,norb)")
    # Match contracted-apply indexing: kernel consumes operator-indexed h[:,p,q] for T_{pq},
    # while SOC convention stores h[:,p_out,q_out] for T_{q_out p_out}.
    h_m_t = cp_mod.swapaxes(h_m_d, 1, 2)
    h_re = cp_mod.ascontiguousarray(h_m_t.real.astype(cp_mod.float64, copy=False))
    h_im = cp_mod.ascontiguousarray(h_m_t.imag.astype(cp_mod.float64, copy=False))

    if out_re is None:
        out_re = cp_mod.zeros((3, nb, nk), dtype=cp_mod.float64)
    else:
        out_re = cp_mod.ascontiguousarray(cp_mod.asarray(out_re, dtype=cp_mod.float64))
        if out_re.shape != (3, nb, nk):
            raise ValueError("out_re must have shape (3,nb,nk)")
        out_re[...] = 0.0
    if out_im is None:
        out_im = cp_mod.zeros((3, nb, nk), dtype=cp_mod.float64)
    else:
        out_im = cp_mod.ascontiguousarray(cp_mod.asarray(out_im, dtype=cp_mod.float64))
        if out_im.shape != (3, nb, nk):
            raise ValueError("out_im must have shape (3,nb,nk)")
        out_im[...] = 0.0

    stream_ptr = _stream_ptr(cp_mod, stream)

    from asuka.cuda.cuda_backend import (  # noqa: PLC0415
        build_epq_action_table_combined_device,
        make_device_drt,
        make_device_state_cache,
    )

    drt_bra_dev = make_device_drt(drt_bra)
    drt_ket_dev = drt_bra_dev if (drt_bra is drt_ket) else make_device_drt(drt_ket)
    ket_state_dev = make_device_state_cache(drt_ket, drt_ket_dev)
    twos_max = _twos_max_for_drts((drt_bra, drt_ket))
    tf_ws = make_triplet_factors_workspace_device(twos_max)

    if bool(use_epq_table_if_possible) and (drt_bra is drt_ket):
        epq_indptr, epq_indices, epq_pq, _epq_data = build_epq_action_table_combined_device(
            drt_ket,
            drt_ket_dev,
            ket_state_dev,
            threads=int(threads),
            sync=True,
            check_overflow=True,
            use_cache=True,
            dtype=np.float64,
            indptr_dtype="int64",
        )
        epq_indices = cp_mod.ascontiguousarray(cp_mod.asarray(epq_indices, dtype=cp_mod.int32))
        epq_pq = cp_mod.ascontiguousarray(cp_mod.asarray(epq_pq, dtype=cp_mod.int32))
        _ext.triplet_build_gm_all_m_from_epq_table_inplace_device(
            drt_ket_dev,
            ket_state_dev,
            epq_indptr,
            epq_indices,
            epq_pq,
            c_bra_d,
            c_ket_d,
            h_re,
            h_im,
            tf_ws,
            out_re,
            out_im,
            int(threads),
            int(stream_ptr),
            bool(sync),
        )
        return out_re, out_im

    _ext.triplet_build_gm_all_m_dfs_inplace_device(
        drt_bra_dev,
        drt_ket_dev,
        ket_state_dev,
        c_bra_d,
        c_ket_d,
        h_re,
        h_im,
        tf_ws,
        out_re,
        out_im,
        int(threads),
        int(stream_ptr),
        bool(sync),
        int(drt_bra.root),
        int(drt_bra.leaf),
        int(drt_bra.twos_target),
        int(drt_ket.twos_target),
    )
    return out_re, out_im
