from __future__ import annotations

"""cuERI-powered DF (density fitting) primitives.

This module provides the minimal building
blocks needed by cuGUGA workflows:
- build whitened AO DF factors `B[μ,ν,Q]`

Callers that still start from PySCF objects should keep the PySCF-specific
packing in separate "bridge" modules.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CuERIDFConfig:
    backend: str = "gpu_rys"
    mode: str = "warp"
    threads: int = 256
    stream: Any | None = None


def build_df_B_from_cueri_packed_bases(
    ao_basis: Any,
    aux_basis: Any,
    *,
    config: CuERIDFConfig | None = None,
    layout: str = "mnQ",
    profile: dict | None = None,
    return_L: bool = False,
):
    """Build whitened DF factors using cuERI DF primitives (GPU).

    Layouts
    - ``layout="mnQ"`` (default): returns ``B[μ,ν,Q]`` with shape ``(nao, nao, naux)``.
    - ``layout="Qmn"``: returns ``BQ[Q,μ,ν]`` with shape ``(naux, nao, nao)``.

    If ``return_L=True``, returns ``(B, L)`` where L is the lower Cholesky
    factor of the regularized aux metric V.  Passing L to the gradient
    function avoids recomputing V on GPU (where 1-ULP kernel non-determinism
    is amplified by ill-conditioned V into gradient errors).
    """

    try:
        import cupy as cp  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for build_df_B_from_cueri_packed_bases") from e

    from asuka.cueri import df as cueri_df  # noqa: PLC0415

    cfg = CuERIDFConfig() if config is None else config

    t0 = None
    if profile is not None:
        import time

        t0 = time.perf_counter()

    V = cueri_df.metric_2c2e_basis(
        aux_basis,
        stream=cfg.stream,
        backend=str(cfg.backend),
        mode=str(cfg.mode),
        threads=int(cfg.threads),
    )
    # Regularize near-singular auxiliary basis metrics.
    # AutoAux bases can have near-linear dependencies that produce tiny
    # negative eigenvalues (e.g. -1e-14), causing Cholesky to fail or
    # silently produce NaN on GPU.  A small diagonal shift fixes this.
    _v_diag = cp.diag(V)
    _v_shift = max(float(cp.max(cp.abs(_v_diag))) * 1e-14, 1e-12)
    V[cp.diag_indices_from(V)] += _v_shift
    L = cp.linalg.cholesky(V)
    X = cueri_df.int3c2e_basis(
        ao_basis,
        aux_basis,
        stream=cfg.stream,
        backend=str(cfg.backend),
        mode=str(cfg.mode),
        threads=int(cfg.threads),
        profile=profile,
    )
    B = cueri_df.whiten_3c2e(X, L)

    layout_s = str(layout).strip().lower()
    if layout_s == "mnq":
        out = B
    elif layout_s == "qmn":
        out = cp.ascontiguousarray(B.transpose((2, 0, 1)))
        # Drop the (nao,nao,naux) view to reduce live memory in callers that
        # only need BQ.
        del B
    else:
        raise ValueError("layout must be one of: 'mnQ', 'Qmn'")

    if profile is not None and t0 is not None:
        cp.cuda.get_current_stream().synchronize()
        import time

        prof = profile.setdefault("cueri_df", {})
        prof["t_build_B_s"] = float(time.perf_counter() - t0)
        try:
            prof["layout"] = str(layout)
            prof["B_shape"] = list(map(int, out.shape))
        except Exception:
            pass

    if return_L:
        return out, L
    return out


__all__ = ["CuERIDFConfig", "build_df_B_from_cueri_packed_bases"]
