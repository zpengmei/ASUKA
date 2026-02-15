from __future__ import annotations

"""Orbital inspection utilities.

This package provides:
- MO/AO composition analysis (Löwdin, Mulliken-style)
- Cube export for visualizing MOs
- Lightweight glue objects for SCF/CASCI/CASSCF outputs
"""

from .inspect import MOInspector

__all__ = ["MOInspector"]

