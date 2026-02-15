from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warnings

from .basis_cart import BasisCartSoA
from .shell_pairs import ShellPairs


@dataclass(frozen=True)
class PairTablesCPU:
    pair_eta: np.ndarray  # float64, shape (total_pair_prims,)
    pair_Px: np.ndarray  # float64, shape (total_pair_prims,)
    pair_Py: np.ndarray  # float64, shape (total_pair_prims,)
    pair_Pz: np.ndarray  # float64, shape (total_pair_prims,)
    pair_cK: np.ndarray  # float64, shape (total_pair_prims,)


def build_pair_tables_cpu(
    basis: BasisCartSoA,
    shell_pairs: ShellPairs,
    *,
    threads: int = 0,
    profile: dict | None = None,
) -> PairTablesCPU:
    """Build primitive-pair tables for all shell pairs (CPU).

    This mirrors the GPU pair-table concept (`DevicePairTables`) but stores arrays on CPU:
      - pair_eta[k]  = a+b
      - pair_P*[k]   = gaussian product center P for that primitive pair
      - pair_cK[k]   = coef_a*coef_b*exp(-mu*|A-B|^2)

    The layout matches `shell_pairs.sp_pair_start/sp_npair` and uses the same primitive-pair
    ordering as the CUDA implementation (idx = ia*nprimB + ib).
    """

    if profile is not None:
        profile.clear()

    try:
        from . import _eri_rys_cpu as _ext  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "CPU ERI extension is not built. Build it with:\n"
            "  python -m asuka.cueri.build_cpu_ext build_ext --inplace"
        ) from e

    threads_i = int(threads)
    if threads_i > 1 and hasattr(_ext, "openmp_enabled") and not bool(_ext.openmp_enabled()):
        warnings.warn(
            "threads>1 requested but asuka.cueri._eri_rys_cpu was built without OpenMP; "
            "rebuild with CUERI_USE_OPENMP=1 to enable parallelism",
            RuntimeWarning,
            stacklevel=2,
        )

    total_pair_prims = int(np.asarray(shell_pairs.sp_pair_start, dtype=np.int64).ravel()[-1])
    pair_eta = np.empty((total_pair_prims,), dtype=np.float64)
    pair_Px = np.empty((total_pair_prims,), dtype=np.float64)
    pair_Py = np.empty((total_pair_prims,), dtype=np.float64)
    pair_Pz = np.empty((total_pair_prims,), dtype=np.float64)
    pair_cK = np.empty((total_pair_prims,), dtype=np.float64)

    t0 = None
    if profile is not None:
        import time

        t0 = time.perf_counter()

    _ext.build_pair_tables_cart_inplace_cpu(
        np.asarray(basis.shell_cxyz, dtype=np.float64, order="C"),
        np.asarray(basis.shell_prim_start, dtype=np.int32, order="C"),
        np.asarray(basis.shell_nprim, dtype=np.int32, order="C"),
        np.asarray(basis.prim_exp, dtype=np.float64, order="C"),
        np.asarray(basis.prim_coef, dtype=np.float64, order="C"),
        np.asarray(shell_pairs.sp_A, dtype=np.int32, order="C"),
        np.asarray(shell_pairs.sp_B, dtype=np.int32, order="C"),
        np.asarray(shell_pairs.sp_pair_start, dtype=np.int32, order="C"),
        np.asarray(shell_pairs.sp_npair, dtype=np.int32, order="C"),
        pair_eta,
        pair_Px,
        pair_Py,
        pair_Pz,
        pair_cK,
        int(threads_i),
    )

    if profile is not None and t0 is not None:
        import time

        profile["t_build_s"] = float(time.perf_counter() - float(t0))
        profile["threads"] = int(threads_i)
        profile["nsp"] = int(shell_pairs.sp_A.shape[0])
        profile["total_pair_prims"] = int(total_pair_prims)

    return PairTablesCPU(
        pair_eta=pair_eta,
        pair_Px=pair_Px,
        pair_Py=pair_Py,
        pair_Pz=pair_Pz,
        pair_cK=pair_cK,
    )


__all__ = ["PairTablesCPU", "build_pair_tables_cpu"]
