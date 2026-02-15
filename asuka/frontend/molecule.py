from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .periodic_table import atomic_mass_amu, atomic_number

_ANGSTROM_TO_BOHR = 1.8897259886


def _parse_atom_string(atom: str) -> list[tuple[str, np.ndarray]]:
    atoms: list[tuple[str, np.ndarray]] = []
    for frag in str(atom).replace("\n", ";").split(";"):
        frag = frag.strip()
        if not frag:
            continue
        tok = frag.split()
        if len(tok) != 4:
            raise ValueError(f"invalid atom fragment: {frag!r} (expected: 'El x y z')")
        sym = tok[0]
        xyz = np.asarray([float(tok[1]), float(tok[2]), float(tok[3])], dtype=np.float64)
        atoms.append((sym, xyz))
    if not atoms:
        raise ValueError("no atoms parsed")
    return atoms


def _parse_atoms(atoms: Any) -> list[tuple[str, np.ndarray]]:
    if isinstance(atoms, str):
        return _parse_atom_string(atoms)
    if isinstance(atoms, (list, tuple)):
        out: list[tuple[str, np.ndarray]] = []
        for item in atoms:
            if isinstance(item, str):
                out.extend(_parse_atom_string(item))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                sym = str(item[0])
                xyz = np.asarray(item[1], dtype=np.float64).reshape((3,))
                out.append((sym, xyz))
            else:
                raise ValueError(f"invalid atom entry: {item!r}")
        if not out:
            raise ValueError("no atoms parsed")
        return out
    raise TypeError("atoms must be a PySCF-like atom string or a list of (sym, (x,y,z))")


@dataclass(frozen=True)
class Molecule:
    """A lightweight molecule container for cuGUGA workflows."""

    atoms_bohr: tuple[tuple[str, np.ndarray], ...]
    charge: int = 0
    spin: int = 0  # nalpha - nbeta
    basis: Any = None  # orbital basis: name string or explicit basis dict
    cart: bool = True
    # Mutable container for attaching analysis artifacts (geomopt, frequencies, etc.).
    # Note: the dataclass is frozen, but the dict contents can be updated.
    results: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_atoms(
        cls,
        atoms: Any,
        *,
        unit: str = "Bohr",
        charge: int = 0,
        spin: int = 0,
        basis: Any = None,
        cart: bool = True,
    ) -> "Molecule":
        atoms_list = _parse_atoms(atoms)
        unit_norm = str(unit).strip().lower()
        if unit_norm in ("bohr", "a0", "au"):
            scale = 1.0
        elif unit_norm in ("angstrom", "ang", "a"):
            scale = _ANGSTROM_TO_BOHR
        else:
            raise ValueError("unit must be 'Bohr' or 'Angstrom'")
        atoms_bohr = tuple((sym, xyz * scale) for sym, xyz in atoms_list)
        return cls(atoms_bohr=atoms_bohr, charge=int(charge), spin=int(spin), basis=basis, cart=bool(cart))

    @property
    def elements(self) -> tuple[str, ...]:
        return tuple(sym for sym, _ in self.atoms_bohr)

    @property
    def natm(self) -> int:
        return int(len(self.atoms_bohr))

    @property
    def coords_bohr(self) -> np.ndarray:
        """Return atomic coordinates as an array (natm,3) in Bohr."""

        return np.asarray([xyz for _sym, xyz in self.atoms_bohr], dtype=np.float64)

    def set_coords_bohr_inplace(self, coords_bohr: np.ndarray) -> None:
        """Update the stored coordinates in-place (Bohr).

        This mutates the underlying numpy arrays in `atoms_bohr`.
        """

        coords = np.asarray(coords_bohr, dtype=np.float64).reshape((-1, 3))
        if coords.shape[0] != int(self.natm):
            raise ValueError("coords_bohr has wrong natm")
        for ia, (_sym, xyz) in enumerate(self.atoms_bohr):
            xyz[:] = coords[ia]

    def atom_symbol(self, i: int) -> str:
        return str(self.atoms_bohr[int(i)][0])

    def atom_mass_list(self) -> list[float]:
        """Return per-atom masses in amu."""

        return [float(atomic_mass_amu(sym)) for sym, _xyz in self.atoms_bohr]

    @property
    def atom(self) -> str:
        # PySCF-like `mol.atom` string (in Bohr) for best-effort interop with
        # utilities that only inspect element symbols.
        parts = [f"{sym} {xyz[0]:.16g} {xyz[1]:.16g} {xyz[2]:.16g}" for sym, xyz in self.atoms_bohr]
        return "; ".join(parts)

    @property
    def nelectron(self) -> int:
        zsum = sum(atomic_number(sym) for sym, _ in self.atoms_bohr)
        return int(zsum - int(self.charge))

    def energy_nuc(self) -> float:
        """Nuclear repulsion energy in Hartree (Bohr geometry)."""

        e = 0.0
        atoms = self.atoms_bohr
        for i in range(len(atoms)):
            Zi = float(atomic_number(atoms[i][0]))
            ri = atoms[i][1]
            for j in range(i + 1, len(atoms)):
                Zj = float(atomic_number(atoms[j][0]))
                rj = atoms[j][1]
                Rij = float(np.linalg.norm(ri - rj))
                if Rij == 0.0:
                    raise ValueError("coincident nuclei")
                e += Zi * Zj / Rij
        return float(e)
    
    def energy_nuc_grad(self) -> np.ndarray:
        """Nuclear repulsion gradient in Hartree/Bohr, shape (natm, 3).

        For E_nuc = Σ_{i<j} Z_i Z_j / |R_i - R_j|, the gradient is
        ∂E/∂R_i = Σ_{j≠i} Z_i Z_j (R_i - R_j) / |R_i - R_j|^3.
        """

        natm = int(self.natm)
        if natm <= 0:
            return np.zeros((0, 3), dtype=np.float64)

        atoms = self.atoms_bohr
        Z = np.asarray([float(atomic_number(sym)) for sym, _xyz in atoms], dtype=np.float64)
        R = np.asarray([xyz for _sym, xyz in atoms], dtype=np.float64).reshape((natm, 3))

        g = np.zeros((natm, 3), dtype=np.float64)
        for i in range(natm):
            for j in range(i + 1, natm):
                rij = R[i] - R[j]
                r = float(np.linalg.norm(rij))
                if r == 0.0:
                    raise ValueError("coincident nuclei")
                fac = float(-Z[i] * Z[j] / (r * r * r))
                g[i] += fac * rij
                g[j] -= fac * rij
        return g

    def as_dict(self) -> dict[str, Any]:
        return {
            "atoms_bohr": [(sym, xyz.tolist()) for sym, xyz in self.atoms_bohr],
            "charge": int(self.charge),
            "spin": int(self.spin),
            "basis": self.basis,
            "cart": bool(self.cart),
        }


__all__ = ["Molecule"]
