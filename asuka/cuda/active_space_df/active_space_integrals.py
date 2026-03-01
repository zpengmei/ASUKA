from __future__ import annotations

"""Builders for GPU active-space 2e integrals via cuERI.

These helpers are intended for the frontend pipeline, where callers
provide packed cuERI AO/aux bases and the current active-space MO coefficients.
"""

import time
from typing import Any

from asuka.integrals.df_integrals import DeviceDFMOIntegrals

from .cueri_builder import CuERIActiveSpaceDFBuilder


def build_device_dfmo_integrals_cueri_df(
    ao_basis: Any,
    aux_basis: Any,
    C_active,
    *,
    backend: str = "gpu_rys",
    aux_block_naux: int = 256,
    max_tile_bytes: int = 256 * 1024 * 1024,
    want_eri_mat: bool = False,
    want_pair_norm: bool = False,
    profile: dict | None = None,
    cached_b_whitened: Any | None = None,
    cache_out: dict | None = None,
) -> DeviceDFMOIntegrals:
    """Build active-space DF integrals (approximate) via cuERI streamed DF.

    Returns a `DeviceDFMOIntegrals` suitable for the GUGA CUDA backend.
    """

    # Fast path: if the caller provides a cached *whitened* AO DF tensor `B`,
    # build the active-space DF objects directly via linear algebra. This avoids
    # rebuilding metric/int3c2e/whitening in Cartesian AO space (which can be
    # >10GB and can OOM 24GB GPUs for larger bases).
    if cached_b_whitened is not None:
        try:
            import cupy as cp
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cached_b_whitened requires CuPy") from e

        from asuka.cueri import df as cueri_df

        t0 = time.perf_counter()
        B_whitened = cp.asarray(cached_b_whitened, dtype=cp.float64)
        if B_whitened.ndim != 3:
            raise ValueError("cached_b_whitened must have shape (nao, nao, naux)")
        if hasattr(B_whitened, "flags") and not bool(B_whitened.flags.c_contiguous):
            B_whitened = cp.ascontiguousarray(B_whitened)

        C_act = cp.asarray(C_active, dtype=cp.float64)
        if C_act.ndim != 2:
            raise ValueError("C_active must have shape (nao, norb)")
        if hasattr(C_act, "flags") and not bool(C_act.flags.c_contiguous):
            C_act = cp.ascontiguousarray(C_act)

        # l_full: (norb*norb, naux)
        l_full = cueri_df.active_Lfull_from_cached_B_whitened(B_whitened, C_act)
        l_full = cp.ascontiguousarray(l_full, dtype=cp.float64)

        norb = int(C_act.shape[1])
        naux_eff = int(l_full.shape[1])

        eri_mat = None
        if bool(want_eri_mat):
            eri_mat = cp.ascontiguousarray(l_full @ l_full.T, dtype=cp.float64)

        # J_{ps} = sum_q (p q| q s) ~= sum_{q,L} d[L,pq] d[L,qs]
        l3 = l_full.reshape(norb, norb, naux_eff)
        j_ps = cp.ascontiguousarray(cp.einsum("pql,qsl->ps", l3, l3, optimize=True), dtype=cp.float64)

        pair_norm = None
        if bool(want_pair_norm):
            pair_norm = cp.ascontiguousarray(cp.linalg.norm(l_full, axis=1), dtype=cp.float64)

        if profile is not None:
            cp.cuda.get_current_stream().synchronize()
            _dt = float(time.perf_counter() - t0)
            profile["t_build_dfmo_s"] = profile.get("t_build_dfmo_s", 0.0) + _dt
            prof = profile.setdefault("cueri_active_df", {})
            prof["cached_B_fast_path"] = True
            prof["t_cached_B_fast_path_s"] = _dt
            prof["norb"] = int(norb)
            prof["naux"] = int(naux_eff)

        return DeviceDFMOIntegrals(
            norb=int(norb),
            l_full=l_full,
            j_ps=j_ps,
            pair_norm=pair_norm,
            eri_mat=eri_mat,
        )

    builder = CuERIActiveSpaceDFBuilder(
        mol=None,
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        backend=str(backend),
        aux_block_naux=int(aux_block_naux),
        max_tile_bytes=int(max_tile_bytes),
    )
    t0 = time.perf_counter()
    dev = builder.build(
        C_active,
        want_eri_mat=bool(want_eri_mat),
        want_j_ps=True,
        want_pair_norm=bool(want_pair_norm),
        out=None,
        profile=profile,
        cached_b_whitened=cached_b_whitened,
        cache_out=cache_out,
    )
    if profile is not None:
        _dt = float(time.perf_counter() - t0)
        profile["t_build_dfmo_s"] = profile.get("t_build_dfmo_s", 0.0) + _dt
    if dev.j_ps is None:  # pragma: no cover
        raise RuntimeError("internal error: expected j_ps to be computed")
    return DeviceDFMOIntegrals(
        norb=int(dev.norb),
        l_full=dev.l_full,
        j_ps=dev.j_ps,
        pair_norm=dev.pair_norm,
        eri_mat=dev.eri_mat,
    )


def build_device_dfmo_integrals_cueri_dense_rys(
    ao_basis: Any,
    C_active,
    *,
    mol: Any | None = None,
    ao_rep: str = "auto",
    builder: Any | None = None,
    threads: int = 256,
    max_tile_bytes: int = 256 * 1024 * 1024,
    eps_ao: float = 0.0,
    profile: dict | None = None,
) -> DeviceDFMOIntegrals:
    """Build active-space **exact** ERIs as an ordered-pair matrix via cuERI generic Rys tiles (GPU)."""

    shape = getattr(C_active, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("C_active must have shape (nao, norb)")
    norb = int(shape[1])
    if norb <= 0:
        raise ValueError("C_active must have norb > 0")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for GPU dense build") from e

    try:
        from asuka.cueri.active_space_dense_gpu import CuERIActiveSpaceDenseGPUBuilder
    except Exception as e:  # pragma: no cover
        raise RuntimeError("cuERI is required for GPU dense build") from e

    from asuka.cueri.eri_utils import j_ps_from_eri_mat

    prof = profile.setdefault("cueri_dense_rys", {}) if profile is not None else None
    builder_reused = builder is not None
    if builder is None:
        builder = CuERIActiveSpaceDenseGPUBuilder(
            mol=mol,
            ao_basis=ao_basis,
            ao_rep=str(ao_rep),
            threads=int(threads),
            max_tile_bytes=int(max_tile_bytes),
            eps_ao=float(eps_ao),
        )
    dev = builder.build(C_active, profile=prof)
    eri_mat = cp.ascontiguousarray(dev.eri_mat, dtype=cp.float64)

    j_ps = j_ps_from_eri_mat(eri_mat, norb=int(norb))
    j_ps = cp.ascontiguousarray(j_ps, dtype=cp.float64)

    if prof is not None:
        cp.cuda.get_current_stream().synchronize()
        prof["norb"] = int(norb)
        prof["eri_mat_nbytes"] = int(getattr(eri_mat, "nbytes", 0))
        prof["ao_rep"] = str(getattr(builder, "ao_rep", ao_rep))
        prof["builder_reused"] = bool(builder_reused)

    return DeviceDFMOIntegrals(
        norb=int(norb),
        l_full=None,
        j_ps=j_ps,
        pair_norm=None,
        eri_mat=eri_mat,
    )


__all__ = [
    "build_device_dfmo_integrals_cueri_dense_rys",
    "build_device_dfmo_integrals_cueri_df",
]
