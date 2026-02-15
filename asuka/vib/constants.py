"""Physical constants used for vibrational analysis.

All constants are in double precision and chosen to be consistent with common
quantum chemistry codes.

Units
-----
- Energy: Hartree (Eh)
- Length: Bohr (a0)
- Mass: atomic mass unit (amu) and atomic units (electron mass)
- Frequency output: wavenumber (cm^-1)

Notes
-----
We keep a small set of constants here to avoid pulling in external
dependencies for simple analysis tasks.
"""

from __future__ import annotations

import math

# Length conversion
BOHR_TO_ANGSTROM = 0.52917721092
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# Mass conversion
# 1 amu in atomic units (electron masses). Value is widely used in QC codes.
AMU_TO_AU = 1822.888486209

# Frequency conversion
# 1 a.u. of angular frequency corresponds to 219474.6313632 cm^-1.
AU_TO_CM1 = 219474.6313632
CM1_TO_AU = 1.0 / AU_TO_CM1

# Boltzmann constant in Hartree/K.
KB_HARTREE_PER_K = 3.166811563e-6

TWOPI = 2.0 * math.pi

__all__ = [
    "BOHR_TO_ANGSTROM",
    "ANGSTROM_TO_BOHR",
    "AMU_TO_AU",
    "AU_TO_CM1",
    "CM1_TO_AU",
    "KB_HARTREE_PER_K",
    "TWOPI",
]
