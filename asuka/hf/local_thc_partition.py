from __future__ import annotations

"""Atom/shell partitioning helpers for local-THC."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class AtomBlocks:
    """Atom blocking result."""

    blocks: list[list[int]]  # list of atom ids per block (in block order)
    atom_to_block: np.ndarray  # int32 (natm,)


def _coords_key(xyz: np.ndarray) -> tuple[float, float, float]:
    x, y, z = map(float, np.asarray(xyz, dtype=np.float64).reshape((3,)))
    return (round(x, 12), round(y, 12), round(z, 12))


def map_shells_to_atoms(shell_cxyz: np.ndarray, atom_coords_bohr: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Map basis shells to atom indices by center coordinates (Bohr).

    Returns
    -------
    shell_to_atom : ndarray[int32], shape (nShell,)
    atom_to_shells : list[list[int]], length natm
    """

    shell_cxyz = np.asarray(shell_cxyz, dtype=np.float64).reshape((-1, 3))
    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64).reshape((-1, 3))
    natm = int(atom_coords_bohr.shape[0])
    nshell = int(shell_cxyz.shape[0])

    if natm <= 0:
        raise ValueError("no atoms")

    atom_map: dict[tuple[float, float, float], int] = {_coords_key(atom_coords_bohr[i]): i for i in range(natm)}

    shell_to_atom = np.empty((nshell,), dtype=np.int32)
    atom_to_shells: list[list[int]] = [[] for _ in range(natm)]

    for ish in range(nshell):
        key = _coords_key(shell_cxyz[ish])
        ia = atom_map.get(key)
        if ia is None:
            # Fallback: pick nearest atom if coordinates are not exactly bitwise-identical.
            d2 = np.sum((atom_coords_bohr - shell_cxyz[ish][None, :]) ** 2, axis=1)
            ia = int(np.argmin(d2))
            if float(d2[ia]) > 1e-16:
                raise ValueError("failed to map shell centers to atom coordinates (unexpected basis centers)")
        shell_to_atom[ish] = int(ia)
        atom_to_shells[int(ia)].append(int(ish))

    return shell_to_atom, atom_to_shells


def build_atom_blocks(coords_bohr: np.ndarray, atom_ao_counts: np.ndarray, *, block_max_ao: int) -> AtomBlocks:
    """Partition atoms into blocks using a simple deterministic PCA-sorted greedy scheme."""

    coords_bohr = np.asarray(coords_bohr, dtype=np.float64).reshape((-1, 3))
    atom_ao_counts = np.asarray(atom_ao_counts, dtype=np.int64).ravel()
    natm = int(coords_bohr.shape[0])
    if natm <= 0:
        return AtomBlocks(blocks=[], atom_to_block=np.zeros((0,), dtype=np.int32))
    if atom_ao_counts.shape != (natm,):
        raise ValueError("atom_ao_counts must have shape (natm,)")

    block_max_ao = int(block_max_ao)
    if block_max_ao <= 0:
        raise ValueError("block_max_ao must be > 0")

    # PCA axis for a stable 1D ordering.
    R = coords_bohr - np.mean(coords_bohr, axis=0, keepdims=True)
    cov = R.T @ R
    try:
        w, v = np.linalg.eigh(cov)
        u = v[:, int(np.argmax(w))]
    except Exception:
        u = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.asarray(u, dtype=np.float64).reshape((3,))
    if not np.all(np.isfinite(u)) or float(np.linalg.norm(u)) == 0.0:
        u = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    u = u / float(np.linalg.norm(u))

    t = coords_bohr @ u
    order = np.argsort(t, kind="stable")

    assigned = np.zeros((natm,), dtype=bool)
    atom_to_block = -np.ones((natm,), dtype=np.int32)
    blocks: list[list[int]] = []

    # Precompute distance matrix for selection (natm is typically small for our initial usage).
    d2 = np.sum((coords_bohr[:, None, :] - coords_bohr[None, :, :]) ** 2, axis=2)

    for a0 in order.tolist():
        if bool(assigned[int(a0)]):
            continue
        block_atoms: list[int] = [int(a0)]
        assigned[int(a0)] = True
        ao_sum = int(atom_ao_counts[int(a0)])

        while True:
            if ao_sum >= int(block_max_ao):
                break

            # Candidate atoms: unassigned.
            cand = np.nonzero(~assigned)[0]
            if int(cand.size) == 0:
                break

            # Choose atom closest to current block; tie-break by PCA coordinate.
            block_arr = np.asarray(block_atoms, dtype=np.int32)
            # min distance to any atom already in block
            min_d2 = np.min(d2[cand[:, None], block_arr[None, :]], axis=1)
            # argmin with stable tie-break: (min_d2, t)
            key = np.lexsort((t[cand], min_d2))
            a = int(cand[int(key[0])])

            ao_a = int(atom_ao_counts[a])
            if ao_sum > 0 and (ao_sum + ao_a) > int(block_max_ao):
                break
            block_atoms.append(a)
            assigned[a] = True
            ao_sum += ao_a

        bid = int(len(blocks))
        for a in block_atoms:
            atom_to_block[int(a)] = int(bid)
        blocks.append(block_atoms)

    if np.any(atom_to_block < 0):
        raise RuntimeError("atom blocking produced unassigned atoms")

    return AtomBlocks(blocks=blocks, atom_to_block=atom_to_block)


def build_atom_neighbors_by_schwarz(
    ao_basis,
    shell_to_atom: np.ndarray,
    *,
    natm: int,
    aux_schwarz_thr: float,
    threads: int = 256,
    mode: str = "warp",
    max_tiles_bytes: int = 64 << 20,
) -> list[set[int]]:
    """Build per-atom neighbor sets based on AO shell-pair Schwarz bounds.

    We mark atoms (a,b) as neighbors if any shell pair between them has
    Schwarz bound Q_AB > aux_schwarz_thr.
    """

    aux_schwarz_thr = float(aux_schwarz_thr)
    if not np.isfinite(aux_schwarz_thr) or aux_schwarz_thr < 0.0:
        raise ValueError("aux_schwarz_thr must be finite and >= 0")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for Schwarz neighbor construction") from e

    from asuka.cueri.shell_pairs import build_shell_pairs_l_order  # noqa: PLC0415
    from asuka.cueri.screening import schwarz_shellpairs_device  # noqa: PLC0415

    shell_to_atom = np.asarray(shell_to_atom, dtype=np.int32).ravel()
    natm = int(natm)
    if natm <= 0:
        return []

    sp = build_shell_pairs_l_order(ao_basis)
    Q_dev = schwarz_shellpairs_device(
        ao_basis,
        sp,
        threads=int(threads),
        mode=str(mode),
        max_tiles_bytes=int(max_tiles_bytes),
    )
    Q = cp.asnumpy(Q_dev).ravel()

    spA = np.asarray(sp.sp_A, dtype=np.int32).ravel()
    spB = np.asarray(sp.sp_B, dtype=np.int32).ravel()
    if spA.shape != spB.shape or spA.shape != Q.shape:
        raise RuntimeError("unexpected shell-pair Schwarz output shapes")

    aA = shell_to_atom[spA].astype(np.int32, copy=False)
    aB = shell_to_atom[spB].astype(np.int32, copy=False)

    mask = Q > float(aux_schwarz_thr)
    aA = aA[mask]
    aB = aB[mask]

    neigh: list[set[int]] = [set() for _ in range(natm)]
    for a, b in zip(aA.tolist(), aB.tolist()):
        neigh[int(a)].add(int(b))
        neigh[int(b)].add(int(a))
    return neigh


def union_neighbors(block_atoms: Iterable[int], neighbors: list[set[int]]) -> list[int]:
    out: set[int] = set(int(a) for a in block_atoms)
    for a in list(out):
        if 0 <= int(a) < int(len(neighbors)):
            out.update(neighbors[int(a)])
    return sorted(out)


__all__ = [
    "AtomBlocks",
    "build_atom_blocks",
    "build_atom_neighbors_by_schwarz",
    "map_shells_to_atoms",
    "union_neighbors",
]

