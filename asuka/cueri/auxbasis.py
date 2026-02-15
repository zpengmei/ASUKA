from __future__ import annotations

from collections.abc import Iterable
import re
import warnings


def _norm_basis_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"\\s+", "", name)
    return name


def infer_jkfit_auxbasis_name(orbital_basis_name: str) -> str:
    """Infer a corresponding JKFIT auxiliary basis name from an orbital basis name.

    This function applies a lightweight, heuristic-based mapping to identify a suitable
    auxiliary basis set for Coulomb/Exchange fitting (JKFIT) based on the provided orbital
    basis name. It covers common basis families used in standard quantum chemistry workflows.

    Parameters
    ----------
    orbital_basis_name : str
        The name of the orbital basis set (e.g., 'def2-svp', 'cc-pvdz').

    Returns
    -------
    str
        The name of the inferred auxiliary basis set.

    Notes
    -----
    - This heuristic matches common families like 'def2', 'cc-pv', 'sto-3g', and Pople sets.
    - If a direct match is not found, a default (typically 'def2-svp-jkfit') is returned with a warning.
    - For element-specific or non-standard basis sets, explicit auxiliary basis specification is recommended.
    """

    name = _norm_basis_name(orbital_basis_name)

    # Already an auxiliary basis.
    if "jkfit" in name:
        return name

    # Def2 family: def2-*-jkfit exists for many standard bases.
    if name.startswith("def2-"):
        return f"{name}-jkfit"

    # Correlation-consistent families (Dunning). Note: this is heuristic and does
    # not guarantee an aux basis exists for every variant.
    if name.startswith("cc-pv") or name.startswith("aug-cc-pv"):
        return f"{name}-jkfit"

    # Minimal bases: PySCF commonly falls back to def2-svp-jkfit.
    if name in ("sto-3g", "sto3g", "minao"):
        return "def2-svp-jkfit"

    # Pople family: no dedicated *-jkfit family is consistently available; use a
    # conservative small JKFIT basis as a practical default.
    if name.startswith(("3-21g", "4-31g", "6-31g", "6-311g")):
        return "cc-pvdz-jkfit"

    # If we don't know, prefer a conservative default but warn loudly.
    warnings.warn(
        f"Could not infer a corresponding JKFIT aux basis for orbital basis '{orbital_basis_name}'. "
        "Falling back to 'def2-svp-jkfit'. Consider passing auxbasis explicitly.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "def2-svp-jkfit"


def infer_jkfit_auxbasis(orbital_basis, elements: Iterable[str]) -> dict[str, str]:
    """Infer a per-element JKFIT auxiliary basis mapping.

    Derives a mapping from element symbols to auxiliary basis set names, appropriate for
    JK fitting, based on the input orbital basis configuration.

    Parameters
    ----------
    orbital_basis : str | dict
        The orbital basis configuration.
        - If `str`: A single basis name applied to all elements.
        - If `dict`: A mapping from element symbols to basis names.
    elements : Iterable[str]
        A collection of element symbols present in the molecule (e.g., ['C', 'H', 'H', 'H', 'H']).

    Returns
    -------
    dict[str, str]
        A dictionary mapping each unique element symbol in `elements` to its inferred
        auxiliary basis name.

    Raises
    ------
    TypeError
        If `orbital_basis` is neither a string nor a dictionary.
    """

    elements = [str(e) for e in elements]
    if isinstance(orbital_basis, str):
        aux = infer_jkfit_auxbasis_name(orbital_basis)
        return {e: aux for e in elements}

    if isinstance(orbital_basis, dict):
        out: dict[str, str] = {}
        for e in elements:
            spec = orbital_basis.get(e)
            if isinstance(spec, str):
                out[e] = infer_jkfit_auxbasis_name(spec)
            else:
                warnings.warn(
                    f"orbital basis for element '{e}' is not a string; cannot infer JKFIT aux basis. "
                    "Falling back to 'def2-svp-jkfit' for this element.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                out[e] = "def2-svp-jkfit"
        return out

    raise TypeError("orbital_basis must be a string or a dict mapping element -> basis name")


__all__ = ["infer_jkfit_auxbasis", "infer_jkfit_auxbasis_name"]

