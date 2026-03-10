from __future__ import annotations

"""Compatibility facade for legacy `asuka.mcscf.sort_mo` imports.

The implementation delegates to the corresponding helpers in PySCF.
"""

from collections.abc import Callable
from typing import Any


def _resolve_addons_function(name: str) -> Callable[..., Any]:
    try:
        from pyscf.mcscf import addons
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "sort_mo helpers require PySCF. Install test/runtime extras, "
            "for example: `pip install -e '.[test]'`."
        ) from exc

    fn = getattr(addons, str(name), None)
    if not callable(fn):  # pragma: no cover
        raise ImportError(f"pyscf.mcscf.addons.{name} is unavailable in this PySCF build")
    return fn


def sort_mo(*args: Any, **kwargs: Any):
    """Delegate to `pyscf.mcscf.addons.sort_mo`."""

    return _resolve_addons_function("sort_mo")(*args, **kwargs)


def sort_mo_by_irrep(*args: Any, **kwargs: Any):
    """Delegate to `pyscf.mcscf.addons.sort_mo_by_irrep`."""

    return _resolve_addons_function("sort_mo_by_irrep")(*args, **kwargs)


__all__ = ["sort_mo", "sort_mo_by_irrep"]
