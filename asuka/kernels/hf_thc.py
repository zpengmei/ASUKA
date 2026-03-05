from __future__ import annotations

"""HF THC CUDA extension kernel registry."""

from typing import Any

from ._base import KernelSymbol, try_import


EXT_MODULE = "asuka._hf_thc_cuda_ext"


def load_ext() -> Any | None:
    """Return the imported HF THC CUDA extension module, or None if unavailable."""

    mod, _err = try_import(EXT_MODULE)
    return mod


def require_ext() -> Any:
    """Return the HF THC CUDA extension module or raise with a build hint."""

    mod, err = try_import(EXT_MODULE)
    if mod is None:
        msg = "HF THC CUDA extension is unavailable"
        if err is not None:
            msg += f" ({type(err).__name__}: {err})"
        msg += ". Build via `python -m asuka.build.hf_thc_cuda_ext`."
        raise RuntimeError(msg)
    return mod


KERNELS: list[KernelSymbol] = [
    KernelSymbol(
        EXT_MODULE,
        "rowwise_dot_f64",
        category="hf_thc",
        purpose="Rowwise dot product helper for THC J builds.",
        io="in: A,X (npt,nao) f64 -> out: m (npt,) f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "scale_rows_f64",
        category="hf_thc",
        purpose="Scale each THC row by a point weight.",
        io="in: X (npt,nao), n (npt,) -> out: scaled (npt,nao) f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "hadamard_inplace_f64",
        category="hf_thc",
        purpose="In-place Hadamard product for THC K blocking.",
        io="in/out: M (npt,nb) f64, in: Z (npt,nb) f64",
    ),
]


def probe() -> dict[str, Any]:
    """Return a structured capability report for the HF THC CUDA extension."""

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
