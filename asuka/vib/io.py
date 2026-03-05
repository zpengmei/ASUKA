from __future__ import annotations

"""I/O utilities for vibrational sampling ensembles."""

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .constants import BOHR_TO_ANGSTROM, AU_TIME_TO_FS


def write_ensemble_xyz(
    path: str | Path,
    symbols: Sequence[str],
    coords: np.ndarray,
    *,
    velocities: np.ndarray | None = None,
    energies: Sequence[float] | None = None,
    comment_prefix: str = "",
) -> None:
    """Write a multi-frame XYZ file from an ensemble of geometries.

    Parameters
    ----------
    path : str or Path
        Output file path.
    symbols : sequence of str
        Atomic symbols, length *natm*.
    coords : ndarray, shape (n_samples, natm, 3)
        Coordinates in Angstrom.
    velocities : ndarray or None, shape (n_samples, natm, 3)
        Velocities in Angstrom/fs.  When provided, vx/vy/vz are appended
        to each atom line.
    energies : sequence of float or None
        Per-sample energies (any unit) written into the comment line.
    comment_prefix : str
        Prefix prepended to each comment line.
    """

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim == 2:
        coords = coords[None, :, :]
    if coords.ndim != 3:
        raise ValueError("coords must have shape (n_samples, natm, 3)")

    n_samples, natm, _ = coords.shape
    symbols = list(symbols)
    if len(symbols) != natm:
        raise ValueError(f"symbols length {len(symbols)} != natm {natm}")

    has_vel = velocities is not None
    if has_vel:
        velocities = np.asarray(velocities, dtype=np.float64)
        if velocities.ndim == 2:
            velocities = velocities[None, :, :]
        if velocities.shape != coords.shape:
            raise ValueError("velocities shape must match coords shape")

    with open(path, "w") as fh:
        for i in range(n_samples):
            fh.write(f"{natm}\n")

            parts = []
            if comment_prefix:
                parts.append(str(comment_prefix))
            parts.append(f"sample={i}")
            if energies is not None:
                parts.append(f"E={float(energies[i]):.10f}")
            fh.write(" ".join(parts) + "\n")

            for ia in range(natm):
                x, y, z = coords[i, ia]
                line = f"{symbols[ia]:<4s} {x:16.10f} {y:16.10f} {z:16.10f}"
                if has_vel:
                    vx, vy, vz = velocities[i, ia]
                    line += f"  {vx:16.10f} {vy:16.10f} {vz:16.10f}"
                fh.write(line + "\n")


def write_wigner_ensemble_xyz(
    path: str | Path,
    mol: Any,
    sample: Any,
    *,
    energies: Sequence[float] | None = None,
    comment_prefix: str = "",
) -> None:
    """Convenience wrapper: write a Wigner ensemble from a Molecule + sample result.

    Accepts either a bare coordinate array (Bohr) or a
    :class:`~asuka.vib.sampling.WignerSample` (Bohr coords, Bohr/au_time
    velocities).  Coordinates and velocities are converted to Angstrom and
    Angstrom/fs for the XYZ file.
    """

    from .sampling import WignerSample  # noqa: PLC0415

    symbols = [sym for sym, _xyz in mol.atoms_bohr]

    if isinstance(sample, WignerSample):
        coords_ang = np.asarray(sample.coords, dtype=np.float64) * float(BOHR_TO_ANGSTROM)
        vel_ang_fs = np.asarray(sample.velocities, dtype=np.float64) * (
            float(BOHR_TO_ANGSTROM) / float(AU_TIME_TO_FS)
        )
        write_ensemble_xyz(
            path,
            symbols,
            coords_ang,
            velocities=vel_ang_fs,
            energies=energies,
            comment_prefix=comment_prefix,
        )
    else:
        coords_ang = np.asarray(sample, dtype=np.float64) * float(BOHR_TO_ANGSTROM)
        write_ensemble_xyz(
            path,
            symbols,
            coords_ang,
            energies=energies,
            comment_prefix=comment_prefix,
        )


__all__ = [
    "write_ensemble_xyz",
    "write_wigner_ensemble_xyz",
]
