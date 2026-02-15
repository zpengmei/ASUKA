from __future__ import annotations

"""Minimal periodic table helpers (symbol ↔ atomic number)."""


_SYMBOLS = (
    # 1-indexed list of element symbols through Og (118).
    None,
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)

_SYMBOL_TO_Z = {s.upper(): i for i, s in enumerate(_SYMBOLS) if s is not None}

# Average atomic masses (amu) through Og (118).
# Values are standard atomic weights (where defined) or conventional mass numbers
# for short-lived elements. These are intended for vibrational analysis and
# other mass-weighted operations (not isotope-resolved work).
_MASSES_AMU: tuple[float | None, ...] = (
    None,
    1.00784,
    4.002602,
    6.94,
    9.0121831,
    10.81,
    12.011,
    14.007,
    15.999,
    18.998403163,
    20.1797,
    22.98976928,
    24.305,
    26.9815385,
    28.085,
    30.973761998,
    32.06,
    35.45,
    39.948,
    39.0983,
    40.078,
    44.955908,
    47.867,
    50.9415,
    51.9961,
    54.938044,
    55.845,
    58.933194,
    58.6934,
    63.546,
    65.38,
    69.723,
    72.63,
    74.921595,
    78.971,
    79.904,
    83.798,
    85.4678,
    87.62,
    88.90584,
    91.224,
    92.90637,
    95.95,
    98.0,
    101.07,
    102.9055,
    106.42,
    107.8682,
    112.414,
    114.818,
    118.71,
    121.76,
    127.6,
    126.90447,
    131.293,
    132.90545196,
    137.327,
    138.90547,
    140.116,
    140.90766,
    144.242,
    145.0,
    150.36,
    151.964,
    157.25,
    158.92535,
    162.5,
    164.93033,
    167.259,
    168.93422,
    173.045,
    174.9668,
    178.49,
    180.94788,
    183.84,
    186.207,
    190.23,
    192.217,
    195.084,
    196.966569,
    200.592,
    204.38,
    207.2,
    208.9804,
    209.0,
    210.0,
    222.0,
    223.0,
    226.0,
    227.0,
    232.0377,
    231.03588,
    238.02891,
    237.0,
    244.0,
    243.0,
    247.0,
    247.0,
    251.0,
    252.0,
    257.0,
    258.0,
    259.0,
    262.0,
    267.0,
    268.0,
    271.0,
    272.0,
    270.0,
    276.0,
    281.0,
    280.0,
    285.0,
    286.0,
    289.0,
    289.0,
    293.0,
    294.0,
    294.0,
)


def atomic_number(symbol: str) -> int:
    sym = str(symbol).strip()
    if not sym:
        raise ValueError("empty element symbol")
    # PySCF (and some Molden readers) may label atoms as e.g. "Li1", "H2".
    # Treat trailing indices as non-semantic and recover the pure element symbol.
    i = 0
    while i < len(sym) and sym[i].isalpha():
        i += 1
    if i > 0:
        sym = sym[:i]
    try:
        return int(_SYMBOL_TO_Z[sym.upper()])
    except KeyError as exc:
        raise ValueError(f"unknown element symbol: {symbol!r}") from exc


def element_symbol(Z: int) -> str:
    Z = int(Z)
    if Z <= 0 or Z >= len(_SYMBOLS) or _SYMBOLS[Z] is None:
        raise ValueError(f"invalid atomic number: {Z}")
    return str(_SYMBOLS[Z])

def atomic_mass_amu(symbol: str) -> float:
    z = atomic_number(symbol)
    m = _MASSES_AMU[z]
    if m is None:  # pragma: no cover
        raise ValueError(f"missing atomic mass for {symbol!r}")
    return float(m)


def element_mass_amu(Z: int) -> float:
    Z = int(Z)
    if Z <= 0 or Z >= len(_MASSES_AMU):
        raise ValueError(f"invalid atomic number: {Z}")
    m = _MASSES_AMU[Z]
    if m is None:  # pragma: no cover
        raise ValueError(f"missing atomic mass for Z={Z}")
    return float(m)


__all__ = ["atomic_number", "element_symbol", "atomic_mass_amu", "element_mass_amu"]
