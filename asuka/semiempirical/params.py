"""Parameter dataclasses and JSON loader for NDDO semiempirical methods."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Conversion factors
EV_TO_HARTREE = 1.0 / 27.211386245988
ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


@dataclass(frozen=True)
class GaussianCoreCorr:
    """AM1/PM3 Gaussian core-core correction parameters.

    k in eV, l in Angstrom^-2, m in Angstrom (MOPAC convention).
    """
    k: float
    l: float
    m: float


@dataclass(frozen=True)
class ElementParams:
    """Per-element NDDO parameters.

    All values stored in atomic units (Hartree, Bohr) after loading.
    """
    Z: int
    uss: float          # One-center one-electron integral U_ss (Hartree)
    upp: float          # One-center one-electron integral U_pp (Hartree)
    beta_s: float       # Resonance integral parameter beta_s (Hartree)
    beta_p: float       # Resonance integral parameter beta_p (Hartree)
    zeta_s: float       # STO exponent for s orbitals (Bohr^-1)
    zeta_p: float       # STO exponent for p orbitals (Bohr^-1)
    gss: float          # (ss|ss) one-center ERI (Hartree)
    gsp: float          # (ss|pp) one-center ERI (Hartree)
    gpp: float          # (pp|pp) one-center ERI (Hartree)
    gp2: float          # (pp|p'p') one-center ERI (Hartree)
    hsp: float          # (sp|sp) one-center ERI (Hartree)
    alpha: float        # Core-core repulsion exponent (Bohr^-1)
    eisol: float        # Isolated atom energy (Hartree)
    gaussians: List[GaussianCoreCorr] = field(default_factory=list)


@dataclass(frozen=True)
class MethodParams:
    """Complete parameter set for a semiempirical method."""
    name: str
    elements: Dict[int, ElementParams]
    is_placeholder: bool = False


def _load_json(path: Path) -> MethodParams:
    """Load parameters from JSON file, converting from MOPAC (eV/Angstrom) to atomic units."""
    with open(path) as f:
        data = json.load(f)

    elements = {}
    for z_str, ep in data["elements"].items():
        Z = int(z_str)
        gaussians = [
            GaussianCoreCorr(k=g["k"], l=g["l"], m=g["m"])
            for g in ep.get("gaussians", [])
        ]
        # Convert eV -> Hartree for energies; Angstrom^-1 -> Bohr^-1 for alpha
        # zeta is already in Bohr^-1
        elements[Z] = ElementParams(
            Z=Z,
            uss=ep["uss"] * EV_TO_HARTREE,
            upp=ep["upp"] * EV_TO_HARTREE,
            beta_s=ep["beta_s"] * EV_TO_HARTREE,
            beta_p=ep["beta_p"] * EV_TO_HARTREE,
            zeta_s=ep["zeta_s"],
            zeta_p=ep["zeta_p"],
            gss=ep["gss"] * EV_TO_HARTREE,
            gsp=ep["gsp"] * EV_TO_HARTREE,
            gpp=ep["gpp"] * EV_TO_HARTREE,
            gp2=ep["gp2"] * EV_TO_HARTREE,
            hsp=ep["hsp"] * EV_TO_HARTREE,
            alpha=ep["alpha"] / ANGSTROM_TO_BOHR,  # Ang^-1 -> Bohr^-1
            eisol=ep["eisol"] * EV_TO_HARTREE,
            gaussians=gaussians,
        )

    return MethodParams(
        name=data["name"],
        elements=elements,
        is_placeholder=bool(data.get("placeholder", False)),
    )


_DATA_DIR = Path(__file__).parent / "data"
_CACHE: Dict[str, MethodParams] = {}


def load_params(method: str = "AM1") -> MethodParams:
    """Load parameters for the specified method.

    Parameters are cached after first load.
    """
    method = method.upper()
    if method not in _CACHE:
        path = _DATA_DIR / f"{method.lower()}_params.json"
        if not path.exists():
            raise FileNotFoundError(f"No parameter file for method '{method}': {path}")
        _CACHE[method] = _load_json(path)
    return _CACHE[method]
