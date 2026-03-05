from __future__ import annotations

"""Orbitals CUDA extension kernel registry.

This wraps the optional `asuka._orbitals_cuda_ext` module, which provides
GPU kernels for:
- cartesian AO value evaluation on point clouds
- density ingredient evaluation
- Becke grid construction helpers
"""

from typing import Any

from ._base import KernelSymbol, try_import


EXT_MODULE = "asuka._orbitals_cuda_ext"


def load_ext() -> Any | None:
    """Return the imported orbitals CUDA extension module, or None if unavailable."""

    mod, _err = try_import(EXT_MODULE)
    return mod


def require_ext() -> Any:
    """Return the orbitals CUDA extension module or raise with a build hint."""

    mod, err = try_import(EXT_MODULE)
    if mod is None:
        msg = "Orbitals CUDA extension is unavailable"
        if err is not None:
            msg += f" ({type(err).__name__}: {err})"
        msg += ". Build via `python -m asuka.build.orbitals_cuda_ext`."
        raise RuntimeError(msg)
    return mod


KERNELS: list[KernelSymbol] = [
    KernelSymbol(
        EXT_MODULE,
        "IngredientsWorkspace",
        category="orbitals_density",
        purpose="CUDA workspace for density ingredient evaluation kernels.",
        io="Python class exported by the extension",
    ),
    KernelSymbol(
        EXT_MODULE,
        "BeckeGridWorkspace",
        category="orbitals_grid",
        purpose="CUDA workspace for Becke atom-block grid kernels.",
        io="Python class exported by the extension",
    ),
    KernelSymbol(
        EXT_MODULE,
        "eval_becke_atom_block_f64_inplace_device",
        category="orbitals_grid",
        purpose="Build Becke atom-block points/weights on device (float64).",
        io="in: atom coords/grid inputs -> out: (npt,3) points + (npt,) weights",
    ),
    KernelSymbol(
        EXT_MODULE,
        "eval_aos_cart_value_f64_inplace_device",
        category="orbitals_aos",
        purpose="Evaluate cartesian AO values on points (float64).",
        io="in: packed shell arrays + points -> out: ao (npt,nao_cart)",
    ),
    KernelSymbol(
        EXT_MODULE,
        "eval_density_otpd_f64_inplace_device",
        category="orbitals_density",
        purpose="Evaluate core/active density ingredients (float64).",
        io="in: packed basis + MO coeffs + dm1/dm2 -> out: rho/pi (+ optional derivs)",
    ),
]


def probe() -> dict[str, Any]:
    """Return a structured capability report for the orbitals CUDA extension."""

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
