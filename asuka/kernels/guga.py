from __future__ import annotations

"""cuGUGA CUDA extension kernel registry.

This wraps the optional native CUDA extension modules:
- `asuka._guga_cuda_ext`: sparse-mv/mm, EPQ table build/apply, CI helpers
- `asuka._guga_cuda_linalg_ext`: dense linalg helpers (cuSOLVER/cuBLAS baseline)
"""

from typing import Any

from ._base import KernelSymbol, try_import


EXT_MODULE = "asuka._guga_cuda_ext"
LINALG_MODULE = "asuka._guga_cuda_linalg_ext"


def load_ext() -> Any | None:
    mod, _err = try_import(EXT_MODULE)
    return mod


def require_ext() -> Any:
    mod, err = try_import(EXT_MODULE)
    if mod is None:
        msg = "cuGUGA CUDA extension is unavailable"
        if err is not None:
            msg += f" ({type(err).__name__}: {err})"
        msg += ". Build via `python -m asuka.build.guga_cuda_ext`."
        raise RuntimeError(msg)
    return mod


def load_linalg_ext() -> Any | None:
    mod, _err = try_import(LINALG_MODULE)
    return mod


def require_linalg_ext() -> Any:
    mod, err = try_import(LINALG_MODULE)
    if mod is None:
        msg = "cuGUGA CUDA linalg extension is unavailable"
        if err is not None:
            msg += f" ({type(err).__name__}: {err})"
        msg += ". Build via `python -m asuka.build.guga_cuda_linalg_ext`."
        raise RuntimeError(msg)
    return mod


KERNELS_CORE: list[KernelSymbol] = [
    KernelSymbol(
        EXT_MODULE,
        "device_info",
        category="guga_meta",
        purpose="Return CUDA device information from the extension.",
    ),
    KernelSymbol(
        EXT_MODULE,
        "mem_info",
        category="guga_meta",
        purpose="Return extension-side memory accounting information.",
    ),
    KernelSymbol(
        EXT_MODULE,
        "ell_spmv_f64_inplace_device",
        category="guga_sparse",
        purpose="ELL sparse matrix-vector multiply (float64).",
    ),
    KernelSymbol(
        EXT_MODULE,
        "ell_spmm_f64_inplace_device",
        category="guga_sparse",
        purpose="ELL sparse matrix-matrix multiply (float64).",
    ),
    KernelSymbol(
        EXT_MODULE,
        "sell_spmv_f64_inplace_device",
        category="guga_sparse",
        purpose="SELL sparse matrix-vector multiply (float64).",
    ),
    KernelSymbol(
        EXT_MODULE,
        "sell_spmm_f64_inplace_device",
        category="guga_sparse",
        purpose="SELL sparse matrix-matrix multiply (float64).",
    ),
    # Key EPQ-table path entrypoints used by `asuka.cuda.cuda_backend.has_*`.
    KernelSymbol(
        EXT_MODULE,
        "epq_contribs_many_count_allpairs_inplace_device",
        category="guga_epq",
        purpose="Build EPQ table on device (count-allpairs).",
    ),
    KernelSymbol(
        EXT_MODULE,
        "epq_contribs_many_count_allpairs_recompute_inplace_device",
        category="guga_epq",
        purpose="Build EPQ table on device (on-the-fly recompute).",
    ),
    KernelSymbol(
        EXT_MODULE,
        "build_t_from_epq_table_inplace_device",
        category="guga_epq",
        purpose="Build T from an EPQ table on device.",
    ),
    KernelSymbol(
        EXT_MODULE,
        "apply_g_flat_gather_epq_table_inplace_device",
        category="guga_epq",
        purpose="Apply G using EPQ table with destination gather on device.",
    ),
    KernelSymbol(
        EXT_MODULE,
        "build_w_from_epq_transpose_range_mm_scaled_inplace_device",
        category="guga_epq",
        purpose="Build W from EPQ transpose-range (mm-scaled) on device.",
    ),
    KernelSymbol(
        EXT_MODULE,
        "build_w_from_epq_transpose_range_mm_inplace_device",
        category="guga_epq",
        purpose="Build W from EPQ transpose-range (mm) on device.",
    ),
]

KERNELS_LINALG: list[KernelSymbol] = [
    KernelSymbol(
        LINALG_MODULE,
        "DenseSymDavidsonWorkspace",
        category="guga_linalg",
        purpose="CUDA workspace for dense symmetric Davidson prototype.",
        io="Python class exported by the extension",
    ),
    KernelSymbol(
        LINALG_MODULE,
        "eigh_sym",
        category="guga_linalg",
        purpose="Symmetric FP64 eigensolve via cuSOLVER.",
    ),
    KernelSymbol(
        LINALG_MODULE,
        "gemm",
        category="guga_linalg",
        purpose="Row-major FP64 GEMM via cuBLAS.",
    ),
    KernelSymbol(
        LINALG_MODULE,
        "davidson_dense_sym",
        category="guga_linalg",
        purpose="Dense symmetric Davidson prototype (validation harness).",
    ),
]


def probe() -> dict[str, Any]:
    """Return a structured capability report for cuGUGA native extensions."""

    core_mod, core_err = try_import(EXT_MODULE)
    linalg_mod, linalg_err = try_import(LINALG_MODULE)

    def _cats(kernels: list[KernelSymbol]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for ks in kernels:
            out.setdefault(str(ks.category), []).append(str(ks.symbol))
        return out

    return {
        "core": {
            "module": str(EXT_MODULE),
            "present": bool(core_mod is not None),
            "import_error": None if core_err is None else f"{type(core_err).__name__}: {core_err}",
            "symbols": {ks.symbol: bool(ks.available()) for ks in KERNELS_CORE},
            "categories": _cats(KERNELS_CORE),
        },
        "linalg": {
            "module": str(LINALG_MODULE),
            "present": bool(linalg_mod is not None),
            "import_error": None if linalg_err is None else f"{type(linalg_err).__name__}: {linalg_err}",
            "symbols": {ks.symbol: bool(ks.available()) for ks in KERNELS_LINALG},
            "categories": _cats(KERNELS_LINALG),
        },
    }


__all__ = [
    "EXT_MODULE",
    "LINALG_MODULE",
    "KERNELS_CORE",
    "KERNELS_LINALG",
    "load_ext",
    "require_ext",
    "load_linalg_ext",
    "require_linalg_ext",
    "probe",
]

