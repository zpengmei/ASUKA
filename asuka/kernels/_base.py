from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Callable


_MODULE_CACHE: dict[str, Any] = {}
_MODULE_ERROR: dict[str, BaseException] = {}


def try_import(module: str) -> tuple[Any | None, BaseException | None]:
    """Best-effort import helper for optional native extensions.

    Returns (module, error). Never raises.
    """

    if module in _MODULE_CACHE:
        return _MODULE_CACHE[module], None
    if module in _MODULE_ERROR:
        return None, _MODULE_ERROR[module]

    try:
        mod = importlib.import_module(str(module))
    except BaseException as e:  # pragma: no cover
        _MODULE_ERROR[module] = e
        return None, e
    _MODULE_CACHE[module] = mod
    return mod, None


@dataclass(frozen=True, slots=True)
class KernelSymbol:
    """Metadata for a single exported kernel symbol."""

    ext_module: str
    symbol: str
    category: str
    purpose: str
    io: str = ""

    def resolve(self) -> Callable[..., Any] | None:
        mod, _err = try_import(self.ext_module)
        if mod is None:
            return None
        fn = getattr(mod, str(self.symbol), None)
        if callable(fn):
            return fn
        return None

    def available(self) -> bool:
        return self.resolve() is not None

