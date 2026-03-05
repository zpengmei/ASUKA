from __future__ import annotations

"""HF DF-JK CUDA extension kernel registry."""

from typing import Any

from ._base import KernelSymbol, try_import


EXT_MODULE = "asuka._hf_df_jk_cuda_ext"


def load_ext() -> Any | None:
    """Return the imported HF DF-JK CUDA extension module, or None if unavailable."""

    mod, _err = try_import(EXT_MODULE)
    return mod


def require_ext() -> Any:
    mod, err = try_import(EXT_MODULE)
    if mod is None:
        msg = "HF DF-JK CUDA extension is unavailable"
        if err is not None:
            msg += f" ({type(err).__name__}: {err})"
        msg += ". Build via `python -m asuka.build.hf_df_jk_cuda_ext` (requires nvcc/cmake/pybind11)."
        raise RuntimeError(msg)
    return mod


KERNELS: list[KernelSymbol] = [
    KernelSymbol(
        EXT_MODULE,
        "DFJKWorkspace",
        category="hf_df_jk",
        purpose="CUDA workspace object for HF DF-JK kernels (methods like k_from_bq_cw, etc.).",
        io="Python class exported by the extension",
    ),
]


def probe() -> dict[str, Any]:
    mod, err = try_import(EXT_MODULE)
    syms = {ks.symbol: bool(ks.available()) for ks in KERNELS}
    return {
        "module": str(EXT_MODULE),
        "present": bool(mod is not None),
        "import_error": None if err is None else f"{type(err).__name__}: {err}",
        "symbols": syms,
    }


__all__ = [
    "EXT_MODULE",
    "KERNELS",
    "load_ext",
    "require_ext",
    "probe",
]

