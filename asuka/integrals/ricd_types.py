from __future__ import annotations

"""Data structures for the RICD (Resolution of Identity Cholesky Decomposition) generator.

These types are used by the RICD auxiliary-basis generator to represent
configuration, intermediate shell data, and the final generated basis.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


# ---------------------------------------------------------------------------
# Public configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RICDOptions:
    """User-facing configuration for the RICD auxiliary-basis generator.

    Attributes
    ----------
    mode : {"acd", "accd"}
        ``"acd"`` keeps the contracted Cholesky shells directly.
        ``"accd"`` (default) additionally compacts the primitive set.
    threshold : float
        Global Cholesky decomposition threshold (Molcas ``Thrshld_CD``).
    skip_high_ac : bool
        If *True*, apply the Molcas ``SHAC`` shell-pair pruning heuristic.
        Default ``False`` corresponds to ``KHAC`` (keep all).
    primitive_threshold_ratio : float
        Ratio ``η`` that tightens the primitive-pool CD threshold relative
        to the per-type contracted threshold (``τ_p = η · τ_t``).
    primitive_retry_halves : int
        Maximum number of threshold-halving retries when the slim primitive
        rank is insufficient.
    renorm_rel_factor : float
        Relative factor used to compute the Cholesky-basis eigenvalue cutoff:
        ``τ_cb = max(renorm_abs_floor, renorm_rel_factor * threshold)``.
    renorm_abs_floor : float
        Absolute floor for the renormalization eigenvalue cutoff.
    type_threshold_scale : Mapping[str, float] | None
        Per-atom-type multiplicative scale applied to *threshold*.
    dccd : bool
        Reserved for future Molcas ``DCCD`` compatibility.
    cache : bool
        Whether the builder should cache generated bases by fingerprint.
    build_backend : {"cpu"}
        Backend for the atomic metric evaluations.  Phase-1 supports CPU only.
    """

    mode: Literal["acd", "accd"] = "accd"
    threshold: float = 1.0e-4
    skip_high_ac: bool = False
    primitive_threshold_ratio: float = 0.2
    primitive_retry_halves: int = 8
    renorm_rel_factor: float = 1.0e-10
    renorm_abs_floor: float = 1.0e-14
    type_threshold_scale: Mapping[str, float] | None = None
    dccd: bool = False
    cache: bool = True
    build_backend: Literal["cpu"] = "cpu"


# ---------------------------------------------------------------------------
# Intermediate shell representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RICDShell:
    """One candidate or generated auxiliary shell in the RICD pipeline.

    Each shell has ``nctr = 1`` in ASUKA's expanded convention.

    Attributes
    ----------
    atom_type_key : str
        Identifier of the atomic basis type this shell belongs to.
    l : int
        Angular momentum.
    prim_exp : Any
        ``np.ndarray[(nprim,), float64]`` — primitive exponents.
    prim_coef : Any
        ``np.ndarray[(nprim,), float64]`` — packed primitive coefficients in
        BasisCartSoA convention.
    """

    atom_type_key: str
    l: int
    prim_exp: Any  # np.ndarray[(nprim,), float64]
    prim_coef: Any  # np.ndarray[(nprim,), float64]


# ---------------------------------------------------------------------------
# Final generated basis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RICDGeneratedBasis:
    """Result of the RICD auxiliary-basis generator.

    Attributes
    ----------
    packed_basis : Any
        ``BasisCartSoA`` — the molecular auxiliary basis ready for DF.
    basis_name : str
        Human-readable summary string (e.g. ``"accd/1e-4"``).
    options : RICDOptions
        The options used for generation.
    stats : dict[str, Any]
        Per-atom-type statistics (shell counts, timings, etc.).
    atom_type_keys : tuple[str, ...]
        Ordered keys of the distinct atomic basis types.
    """

    packed_basis: Any  # BasisCartSoA
    basis_name: str
    options: RICDOptions
    stats: dict[str, Any]
    atom_type_keys: tuple[str, ...]


# ---------------------------------------------------------------------------
# Internal helpers (not part of the public API)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtomicBasisType:
    """Unique atomic AO basis type, grouping symmetry-equivalent atoms.

    Attributes
    ----------
    key : str
        Fingerprint string that uniquely identifies this basis type
        (e.g. ``"C_6s3p1d"``).
    symbol : str
        Element symbol.
    atomic_number : int
        Atomic number.
    atom_indices : tuple[int, ...]
        Indices of atoms in the molecule that share this basis type.
    rep_center : Any
        ``np.ndarray[(3,), float64]`` — representative center (first atom).
    local_shell_meta : tuple[tuple[int, int, int, int], ...]
        Per-local-shell metadata: ``(local_idx, l, nprim, prim_start)``
        where ``prim_start`` is the primitive offset in the molecular basis.
    """

    key: str
    symbol: str
    atomic_number: int
    atom_indices: tuple[int, ...]
    rep_center: Any  # np.ndarray[(3,), float64]
    local_shell_meta: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class ShellBlock:
    """Metadata for one shell block inside the atomic candidate metric.

    Attributes
    ----------
    shell_id : int
        Index of the candidate shell in the pool list.
    l : int
        Angular momentum of the shell.
    dim : int
        Number of Cartesian functions: ``ncart(l)``.
    offset : int
        Starting row/column in the metric matrix.
    """

    shell_id: int
    l: int
    dim: int
    offset: int


# ---------------------------------------------------------------------------
# Helpers for RICD option normalization
# ---------------------------------------------------------------------------

_RICD_AUXBASIS_ALIASES: frozenset[str] = frozenset({"ricd", "acd", "accd"})


def is_ricd_request(auxbasis: Any, ricd_options: Any = None) -> bool:
    """Return *True* if *auxbasis* (or *ricd_options*) requests RICD generation."""
    if isinstance(auxbasis, RICDOptions):
        return True
    if isinstance(ricd_options, RICDOptions):
        return True
    if isinstance(auxbasis, str) and str(auxbasis).strip().lower() in _RICD_AUXBASIS_ALIASES:
        return True
    return False


def normalize_ricd_options(auxbasis: Any, ricd_options: Any = None) -> RICDOptions:
    """Resolve *auxbasis* / *ricd_options* into a concrete :class:`RICDOptions`."""
    if isinstance(ricd_options, RICDOptions):
        return ricd_options
    if isinstance(auxbasis, RICDOptions):
        return auxbasis
    if isinstance(auxbasis, str):
        tag = str(auxbasis).strip().lower()
        if tag == "acd":
            return RICDOptions(mode="acd")
        # "ricd" and "accd" both map to accd (default)
        return RICDOptions(mode="accd")
    raise TypeError(f"cannot resolve RICD options from auxbasis={auxbasis!r}")


__all__ = [
    "AtomicBasisType",
    "RICDGeneratedBasis",
    "RICDOptions",
    "RICDShell",
    "ShellBlock",
    "is_ricd_request",
    "normalize_ricd_options",
]
