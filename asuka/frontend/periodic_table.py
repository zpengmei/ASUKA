from __future__ import annotations

"""Compatibility facade for periodic-table helpers.

Preferred import path for shared runtime use is `asuka.chem.periodic_table`.
"""

from asuka.chem.periodic_table import atomic_mass_amu, atomic_number

__all__ = ["atomic_number", "atomic_mass_amu"]
