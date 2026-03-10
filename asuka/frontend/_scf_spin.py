from __future__ import annotations


def nalpha_nbeta_from_mol(mol) -> tuple[int, int]:
    nelec = int(mol.nelectron)
    spin = int(mol.spin)
    if nelec <= 0:
        raise ValueError("nelectron must be positive")
    if (nelec + spin) % 2 != 0 or (nelec - spin) % 2 != 0:
        raise ValueError("incompatible nelectron/spin parity (requires nelec±spin even)")
    nalpha = (nelec + spin) // 2
    nbeta = (nelec - spin) // 2
    if nalpha < 0 or nbeta < 0:
        raise ValueError("invalid nelectron/spin combination (negative nalpha/nbeta)")
    return int(nalpha), int(nbeta)
