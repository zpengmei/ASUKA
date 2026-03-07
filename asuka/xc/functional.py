from __future__ import annotations

"""Functional specification and registry for Minnesota meta-GGA functionals."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FunctionalSpec:
    """Specification for a DFT exchange-correlation functional."""

    name: str
    cx_hf: float  # fraction of exact (HF) exchange
    needs_tau: bool  # True for meta-GGA


_REGISTRY: dict[str, FunctionalSpec] = {
    "mn15": FunctionalSpec(name="mn15", cx_hf=0.44, needs_tau=True),
    "m06": FunctionalSpec(name="m06", cx_hf=0.27, needs_tau=True),
    "m06-2x": FunctionalSpec(name="m06-2x", cx_hf=0.54, needs_tau=True),
    "m06-l": FunctionalSpec(name="m06-l", cx_hf=0.0, needs_tau=True),
}


def get_functional(name: str) -> FunctionalSpec:
    """Look up a functional by name (case-insensitive)."""
    key = str(name).strip().lower()
    spec = _REGISTRY.get(key)
    if spec is None:
        raise ValueError(f"Unknown functional '{name}'. Available: {sorted(_REGISTRY)}")
    return spec


__all__ = ["FunctionalSpec", "get_functional"]
