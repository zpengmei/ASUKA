"""Subspace classification for SST-CASPT2.

Maps IC-CASPT2 cases (1-13) to the leading / supporting / trailing
subspace partition used in the SST factorization (Song 2024, Section III.A).

Leading subspace (L):
    Cases that involve BOTH core AND virtual indices.
    These are fully captured by E^MP2_dressed.
    IC cases: E±(6-7), G±(10-11), H±(12-13)

Supporting subspace (S):
    Cases that involve ONLY core OR ONLY virtual (not both).
    These require the Löwdin partition correction.
    IC cases: A(1), B±(2-3), C(4), D(5), F±(8-9)

Trailing subspace (T):
    A subset of the supporting cases that retains non-trivial solutions
    after the G_apx approximation. Determined dynamically based on
    the specific system.
"""
from __future__ import annotations

from typing import FrozenSet

__all__ = [
    "LEADING_CASES",
    "SUPPORT_CASES",
    "ALL_CASES",
    "CASE_NAMES",
    "is_leading",
    "is_supporting",
]

# IC case numbers (1-based, matching ASUKA convention)
LEADING_CASES: FrozenSet[int] = frozenset({6, 7, 10, 11, 12, 13})
SUPPORT_CASES: FrozenSet[int] = frozenset({1, 2, 3, 4, 5, 8, 9})
ALL_CASES: FrozenSet[int] = LEADING_CASES | SUPPORT_CASES

CASE_NAMES = {
    1: "A(VJTU)",
    2: "B+(VJTIP)",
    3: "B-(VJTIM)",
    4: "C(ATVX)",
    5: "D(AIVX)",
    6: "E+(VJAIP)",
    7: "E-(VJAIM)",
    8: "F+(BVATP)",
    9: "F-(BVATM)",
    10: "G+(BJATQ)",
    11: "G-(BJATM)",
    12: "H+(BJAIP)",
    13: "H-(BJAIM)",
}


def is_leading(case: int) -> bool:
    """Check if an IC case belongs to the leading subspace."""
    return case in LEADING_CASES


def is_supporting(case: int) -> bool:
    """Check if an IC case belongs to the supporting subspace."""
    return case in SUPPORT_CASES
