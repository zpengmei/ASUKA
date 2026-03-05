from __future__ import annotations

"""CASPT2 CUDA extension kernel registry."""

from typing import Any

from ._base import KernelSymbol, try_import


EXT_MODULE = "asuka._caspt2_cuda_ext"


def load_ext() -> Any | None:
    """Return the imported CASPT2 CUDA extension module, or None if unavailable."""

    mod, _err = try_import(EXT_MODULE)
    return mod


def require_ext() -> Any:
    """Return the CASPT2 CUDA extension module or raise with a build hint."""

    mod, err = try_import(EXT_MODULE)
    if mod is None:
        msg = "CASPT2 CUDA extension is unavailable"
        if err is not None:
            msg += f" ({type(err).__name__}: {err})"
        msg += ". Build via `python -m asuka.build.caspt2_cuda_ext`."
        raise RuntimeError(msg)
    return mod


KERNELS: list[KernelSymbol] = [
    KernelSymbol(
        EXT_MODULE,
        "apply_h0diag_sr_f64",
        category="caspt2_diag",
        purpose="Apply the shifted H0 diagonal operator for SR systems.",
    ),
    KernelSymbol(
        EXT_MODULE,
        "apply_precond_sr_f64",
        category="caspt2_diag",
        purpose="Apply the SR preconditioner on device.",
    ),
    KernelSymbol(
        EXT_MODULE,
        "mltsca_f64",
        category="caspt2_listops",
        purpose="List-driven scaled accumulation kernel.",
    ),
    KernelSymbol(
        EXT_MODULE,
        "mltdxp_f64",
        category="caspt2_listops",
        purpose="List-driven dyadic contraction kernel.",
    ),
    KernelSymbol(
        EXT_MODULE,
        "mltmv_f64",
        category="caspt2_listops",
        purpose="List-driven matvec-like contraction kernel.",
    ),
    KernelSymbol(
        EXT_MODULE,
        "mltr1_f64",
        category="caspt2_listops",
        purpose="List-driven rank-1 contraction kernel.",
    ),
    KernelSymbol(
        EXT_MODULE,
        "ddot_f64",
        category="caspt2_blas",
        purpose="Device dot product helper used by CASPT2 CUDA solvers.",
    ),
]


def probe() -> dict[str, Any]:
    """Return a structured capability report for the CASPT2 CUDA extension."""

    mod, err = try_import(EXT_MODULE)
    syms = {ks.symbol: bool(ks.available()) for ks in KERNELS}
    cats: dict[str, list[str]] = {}
    for ks in KERNELS:
        cats.setdefault(str(ks.category), []).append(str(ks.symbol))
    return {
        "module": str(EXT_MODULE),
        "present": bool(mod is not None),
        "import_error": None if err is None else f"{type(err).__name__}: {err}",
        "symbols": syms,
        "categories": cats,
    }


__all__ = [
    "EXT_MODULE",
    "KERNELS",
    "load_ext",
    "require_ext",
    "probe",
]
