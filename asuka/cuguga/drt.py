from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

STEP_ORDER: tuple[str, ...] = ("E", "U", "L", "D")
STEP_TO_INDEX: dict[str, int] = {s: i for i, s in enumerate(STEP_ORDER)}


@dataclass(eq=False)
class DRT:
    """Distinct Row Table (DRT) for a GUGA CSF space.

    The DRT is a directed acyclic graph encoding all Configuration State
    Functions (CSFs) for a given ``(norb, nelec, 2S)`` specification.
    Each CSF corresponds to a root-to-leaf walk through the graph, where
    the four step types E (empty), U (up), L (down), D (double) map to
    orbital occupancies 0, 1, 1, 2 respectively.

    Attributes
    ----------
    norb : int
        Number of spatial orbitals.
    nelec : int
        Number of electrons.
    twos_target : int
        Target spin: twice the total spin quantum number (2S).
    node_k : np.ndarray
        int16 array of orbital index at each node.
    node_ne : np.ndarray
        int16 array of cumulative electron count at each node.
    node_twos : np.ndarray
        int16 array of cumulative spin (2S) at each node.
    node_sym : np.ndarray
        int16 array of cumulative spatial irrep at each node.
    nwalks : np.ndarray
        int64 array of walk counts (number of paths through each node).
    child : np.ndarray
        int32 array of shape ``(nnodes, 4)`` giving child node IDs for
        each step type (E=0, U=1, L=2, D=3). Missing children are ``-1``.
    root : int
        Root node ID (always 0).
    leaf : int
        Leaf node ID.
    ncsf : int
        Total number of CSFs (walks from root to leaf).
    """

    norb: int
    nelec: int
    twos_target: int

    node_k: np.ndarray
    node_ne: np.ndarray
    node_twos: np.ndarray
    node_sym: np.ndarray
    nwalks: np.ndarray
    child: np.ndarray
    root: int
    leaf: int
    ncsf: int

    def __post_init__(self) -> None:
        if self.child.shape != (self.node_k.size, len(STEP_ORDER)):
            raise ValueError("child table has incompatible shape")
        if self.node_sym.shape != self.node_k.shape:
            raise ValueError("node_sym has incompatible shape")
        if self.root < 0 or self.root >= self.node_k.size:
            raise ValueError("invalid root node id")
        if self.leaf < 0 or self.leaf >= self.node_k.size:
            raise ValueError("invalid leaf node id")

    @property
    def nnodes(self) -> int:
        """Total number of nodes in the DRT graph."""
        return int(self.node_k.size)

    def node_state(self, node_id: int) -> tuple[int, int, int]:
        """Return the ``(k, ne, twos)`` state tuple for a node.

        Parameters
        ----------
        node_id : int
            Node index.

        Returns
        -------
        tuple of (int, int, int)
            Orbital index, electron count, and spin at this node.
        """
        return (
            int(self.node_k[node_id]),
            int(self.node_ne[node_id]),
            int(self.node_twos[node_id]),
        )

    def node_irrep(self, node_id: int) -> int:
        """Return the spatial irrep index for a node.

        Parameters
        ----------
        node_id : int
            Node index.

        Returns
        -------
        int
            Spatial irrep (XOR-product of singly-occupied orbital irreps).
        """
        return int(self.node_sym[node_id])

    def index_to_path(self, idx: int) -> np.ndarray:
        """Convert a CSF index to its step-vector path through the DRT.

        Parameters
        ----------
        idx : int
            CSF index in ``[0, ncsf)``.

        Returns
        -------
        np.ndarray
            int8 array of length ``norb`` with step indices
            (0=E, 1=U, 2=L, 3=D).

        Raises
        ------
        IndexError
            If ``idx`` is out of range.
        """
        idx = int(idx)
        if idx < 0 or idx >= self.ncsf:
            raise IndexError(f"CSF index out of range: {idx} (ncsf={self.ncsf})")

        steps = np.empty(self.norb, dtype=np.int8)
        node = self.root
        for k in range(self.norb):
            for sidx, _s in enumerate(STEP_ORDER):
                child = int(self.child[node, sidx])
                if child < 0:
                    continue
                w = int(self.nwalks[child])
                if idx < w:
                    steps[k] = sidx
                    node = child
                    break
                idx -= w
            else:
                raise RuntimeError("index_to_path failed to select a step")

        if node != self.leaf:
            raise RuntimeError("index_to_path did not terminate at target leaf")
        return steps

    def path_to_index(self, steps: Sequence[int | str]) -> int:
        """Convert a step-vector path to its CSF index.

        Parameters
        ----------
        steps : sequence of int or str
            Step vector of length ``norb``. Each element is either an
            integer index (0-3) or a letter (``"E"``, ``"U"``, ``"L"``,
            ``"D"``).

        Returns
        -------
        int
            CSF index in ``[0, ncsf)``.

        Raises
        ------
        ValueError
            If the path is invalid or has wrong length.
        """
        if len(steps) != self.norb:
            raise ValueError(f"expected path length {self.norb}, got {len(steps)}")

        idx = 0
        node = self.root
        for step in steps:
            if isinstance(step, str):
                sidx = STEP_TO_INDEX.get(step)
                if sidx is None:
                    raise ValueError(f"invalid step {step!r}")
            else:
                sidx = int(step)
                if sidx < 0 or sidx >= len(STEP_ORDER):
                    raise ValueError(f"invalid step index {sidx}")

            for prior in range(sidx):
                child = int(self.child[node, prior])
                if child >= 0:
                    idx += int(self.nwalks[child])

            child = int(self.child[node, sidx])
            if child < 0:
                raise ValueError("invalid path: forbidden step at node")
            node = child

        if node != self.leaf:
            raise ValueError("invalid path: does not reach target leaf")
        return int(idx)

    def path_as_str(self, steps: Sequence[int | str]) -> str:
        """Format a step vector as a human-readable string like ``"EULD..."``.

        Parameters
        ----------
        steps : sequence of int or str
            Step vector of length ``norb``.

        Returns
        -------
        str
            Concatenation of step letters.
        """
        out: list[str] = []
        for step in steps:
            if isinstance(step, str):
                out.append(step)
            else:
                out.append(STEP_ORDER[int(step)])
        return "".join(out)

    def path_to_occ(self, steps: Sequence[int | str]) -> np.ndarray:
        """Convert a step vector to orbital occupancies.

        Parameters
        ----------
        steps : sequence of int or str
            Step vector of length ``norb``.

        Returns
        -------
        np.ndarray
            int8 array of occupancies (0, 1, or 2) per orbital.
        """
        occ = np.empty(self.norb, dtype=np.int8)
        for i, step in enumerate(steps):
            if isinstance(step, str):
                s = step
            else:
                s = STEP_ORDER[int(step)]
            if s == "E":
                occ[i] = 0
            elif s in ("U", "L"):
                occ[i] = 1
            elif s == "D":
                occ[i] = 2
            else:
                raise ValueError(f"invalid step {s!r}")
        return occ

    def path_to_twos_prefix(self, steps: Sequence[int | str]) -> np.ndarray:
        """Compute the cumulative spin prefix along a step vector.

        Parameters
        ----------
        steps : sequence of int or str
            Step vector of length ``norb``.

        Returns
        -------
        np.ndarray
            int16 array of length ``norb + 1`` where element ``k`` is the
            cumulative 2S value after processing orbitals ``0..k-1``.
        """
        twos = np.empty(self.norb + 1, dtype=np.int16)
        twos[0] = 0
        cur = 0
        for i, step in enumerate(steps):
            if isinstance(step, str):
                s = step
            else:
                s = STEP_ORDER[int(step)]
            if s == "U":
                cur += 1
            elif s == "L":
                cur -= 1
            twos[i + 1] = cur
        return twos

    def iter_nodes_sorted(self) -> Iterable[int]:
        """Iterate node IDs in ``(k, ne, twos, sym)`` lexicographic order.

        Returns
        -------
        Iterable[int]
            Node IDs in sorted order.
        """
        order = np.lexsort((self.node_sym, self.node_twos, self.node_ne, self.node_k))
        return (int(i) for i in order)


def _validate_inputs(norb: int, nelec: int, twos_target: int) -> None:
    if norb < 0:
        raise ValueError("norb must be >= 0")
    if nelec < 0:
        raise ValueError("nelec must be >= 0")
    if nelec > 2 * norb:
        raise ValueError("nelec must be <= 2*norb")
    if twos_target < 0:
        raise ValueError("twos_target must be >= 0")
    if (nelec - twos_target) % 2 != 0:
        raise ValueError("nelec and twos_target must have the same parity")
    if twos_target > nelec:
        raise ValueError("twos_target must be <= nelec")


def _normalize_orbsym(norb: int, orbsym: Sequence[int] | None) -> np.ndarray | None:
    if orbsym is None:
        return None
    arr = np.asarray(orbsym, dtype=np.int32).ravel()
    if int(arr.size) != int(norb):
        raise ValueError(f"orbsym has wrong length: {int(arr.size)} (expected {int(norb)})")
    if np.any(arr < 0):
        raise ValueError("orbsym entries must be >= 0")
    return arr


def _sym_nbits(orbsym: np.ndarray, wfnsym: int) -> int:
    max_sym = int(wfnsym)
    if int(orbsym.size):
        max_sym = max(max_sym, int(orbsym.max()))
    return int(max_sym).bit_length()


def _sym_mul(a: int, b: int) -> int:
    # PySCF uses bit-coded irreps for abelian point groups; multiplication is XOR.
    return int(a) ^ int(b)


def build_drt(
    norb: int,
    nelec: int,
    twos_target: int,
    *,
    orbsym: Sequence[int] | None = None,
    wfnsym: int | None = None,
    ne_constraints: dict[int, tuple[int, int]] | None = None,
) -> DRT:
    """Build a Distinct Row Table enumerating all CSFs for the given space.

    The algorithm uses dynamic programming (backward pass) to count walks
    through the ``(k, ne, twos, sym)`` state space, followed by a BFS
    (forward pass) to construct the explicit graph nodes and child links.

    Parameters
    ----------
    norb : int
        Number of spatial orbitals.
    nelec : int
        Number of electrons.
    twos_target : int
        Target spin: twice the total spin quantum number (2S).
    orbsym : sequence of int, optional
        Bit-coded spatial irreps for each orbital (abelian point groups;
        multiplication is XOR). Required together with ``wfnsym``.
    wfnsym : int, optional
        Target wavefunction spatial irrep. Required together with ``orbsym``.
    ne_constraints : dict[int, tuple[int, int]], optional
        Electron count constraints per orbital index. Maps orbital index
        ``k`` to ``(ne_min, ne_max)`` bounds on the cumulative electron
        count at that node. Useful for GAS/RAS-style restrictions.

    Returns
    -------
    DRT
        The constructed Distinct Row Table.

    Examples
    --------
    >>> from asuka.cuguga import build_drt
    >>> drt = build_drt(norb=4, nelec=4, twos_target=0)
    >>> drt.ncsf
    20
    """
    norb = int(norb)
    nelec = int(nelec)
    twos_target = int(twos_target)
    _validate_inputs(norb, nelec, twos_target)

    orbsym_arr = _normalize_orbsym(norb, orbsym)
    use_sym = (orbsym_arr is not None) and (wfnsym is not None)
    if not use_sym:
        wfnsym_int = 0
        nsym = 1
        orbsym_arr = None
    else:
        wfnsym_int = int(wfnsym)
        if wfnsym_int < 0:
            raise ValueError("wfnsym must be >= 0")
        nbits = _sym_nbits(orbsym_arr, wfnsym_int)
        nsym = 1 << nbits
        if wfnsym_int >= nsym:
            raise ValueError(f"wfnsym={wfnsym_int} is out of range for nsym={nsym}")
        if np.any(orbsym_arr >= nsym):
            raise ValueError(f"orbsym entries must be < nsym={nsym}")

    if ne_constraints is None:
        ne_constraints_norm = None
    else:
        ne_constraints_norm: dict[int, tuple[int, int]] = {}
        for k, bounds in dict(ne_constraints).items():
            kk = int(k)
            if kk < 0 or kk > norb:
                raise ValueError(f"ne_constraints has invalid k={kk} (expected 0 <= k <= {norb})")
            if bounds is None or len(bounds) != 2:
                raise ValueError(f"ne_constraints[{kk}] must be a (ne_min, ne_max) tuple")
            ne_min = int(bounds[0])
            ne_max = int(bounds[1])
            if ne_min < 0 or ne_max < 0:
                raise ValueError(f"ne_constraints[{kk}] bounds must be >= 0")
            if ne_min > ne_max:
                raise ValueError(f"ne_constraints[{kk}] must satisfy ne_min <= ne_max")
            if ne_max > nelec:
                raise ValueError(f"ne_constraints[{kk}] ne_max={ne_max} exceeds nelec={nelec}")
            ne_constraints_norm[kk] = (ne_min, ne_max)

    b = np.zeros((norb + 1, nelec + 1, nelec + 1, nsym), dtype=np.int64)
    b[norb, nelec, twos_target, wfnsym_int] = 1
    if ne_constraints_norm is not None and norb in ne_constraints_norm:
        ne_min, ne_max = ne_constraints_norm[norb]
        if ne_min > 0:
            b[norb, :ne_min, :, :] = 0
        if ne_max < nelec:
            b[norb, ne_max + 1 :, :, :] = 0
    for k in range(norb - 1, -1, -1):
        sym_orb = 0 if orbsym_arr is None else int(orbsym_arr[k])
        for ne in range(nelec, -1, -1):
            for twos in range(nelec, -1, -1):
                for sym in range(nsym):
                    w = b[k + 1, ne, twos, sym]  # E
                    if ne + 1 <= nelec and twos + 1 <= nelec:
                        w += b[k + 1, ne + 1, twos + 1, _sym_mul(sym, sym_orb)]  # U
                    if ne + 1 <= nelec and twos >= 1:
                        w += b[k + 1, ne + 1, twos - 1, _sym_mul(sym, sym_orb)]  # L
                    if ne + 2 <= nelec:
                        w += b[k + 1, ne + 2, twos, sym]  # D
                    b[k, ne, twos, sym] = w
        if ne_constraints_norm is not None and k in ne_constraints_norm:
            ne_min, ne_max = ne_constraints_norm[k]
            if ne_min > 0:
                b[k, :ne_min, :, :] = 0
            if ne_max < nelec:
                b[k, ne_max + 1 :, :, :] = 0

    ncsf = int(b[0, 0, 0, 0])
    if ncsf == 0:
        node_k = np.asarray([0], dtype=np.int16)
        node_ne = np.asarray([0], dtype=np.int16)
        node_twos = np.asarray([0], dtype=np.int16)
        node_sym = np.asarray([0], dtype=np.int16)
        nwalks = np.asarray([0], dtype=np.int64)
        child = np.full((1, len(STEP_ORDER)), -1, dtype=np.int32)
        return DRT(
            norb=norb,
            nelec=nelec,
            twos_target=twos_target,
            node_k=node_k,
            node_ne=node_ne,
            node_twos=node_twos,
            node_sym=node_sym,
            nwalks=nwalks,
            child=child,
            root=0,
            leaf=0,
            ncsf=0,
        )

    root_state = (0, 0, 0, 0)

    state_to_id: dict[tuple[int, int, int, int], int] = {root_state: 0}
    node_k: list[int] = [0]
    node_ne: list[int] = [0]
    node_twos: list[int] = [0]
    node_sym: list[int] = [0]
    nwalks: list[int] = [int(b[0, 0, 0, 0])]
    child_rows: list[list[int]] = [[-1] * len(STEP_ORDER)]

    queue: deque[tuple[int, int, int, int]] = deque([root_state])
    while queue:
        k, ne, twos, sym = queue.popleft()
        node_id = state_to_id[(k, ne, twos, sym)]

        sym_orb = 0 if (orbsym_arr is None or k >= norb) else int(orbsym_arr[k])

        candidates: list[tuple[int, int, int, int]] = []
        candidates.append((k + 1, ne, twos, sym))  # E
        candidates.append((k + 1, ne + 1, twos + 1, _sym_mul(sym, sym_orb)))  # U
        candidates.append((k + 1, ne + 1, twos - 1, _sym_mul(sym, sym_orb)))  # L
        candidates.append((k + 1, ne + 2, twos, sym))  # D

        for sidx, (ck, cne, ctwos, csym) in enumerate(candidates):
            if ck > norb:
                continue
            if cne < 0 or cne > nelec:
                continue
            if ctwos < 0 or ctwos > nelec:
                continue
            if csym < 0 or csym >= nsym:
                continue
            if b[ck, cne, ctwos, csym] == 0:
                continue

            child_id = state_to_id.get((ck, cne, ctwos, csym))
            if child_id is None:
                child_id = len(node_k)
                state_to_id[(ck, cne, ctwos, csym)] = child_id
                node_k.append(int(ck))
                node_ne.append(int(cne))
                node_twos.append(int(ctwos))
                node_sym.append(int(csym))
                nwalks.append(int(b[ck, cne, ctwos, csym]))
                child_rows.append([-1] * len(STEP_ORDER))
                queue.append((ck, cne, ctwos, csym))

            child_rows[node_id][sidx] = int(child_id)

    leaf_id = state_to_id[(norb, nelec, twos_target, wfnsym_int)]
    return DRT(
        norb=norb,
        nelec=nelec,
        twos_target=twos_target,
        node_k=np.asarray(node_k, dtype=np.int16),
        node_ne=np.asarray(node_ne, dtype=np.int16),
        node_twos=np.asarray(node_twos, dtype=np.int16),
        node_sym=np.asarray(node_sym, dtype=np.int16),
        nwalks=np.asarray(nwalks, dtype=np.int64),
        child=np.asarray(child_rows, dtype=np.int32),
        root=0,
        leaf=int(leaf_id),
        ncsf=int(ncsf),
    )


def build_drt_symm(
    norb: int,
    nelec: int,
    twos_target: int,
    *,
    orbsym: Sequence[int],
    wfnsym: int,
) -> DRT:
    """Build a symmetry-filtered DRT for a target spatial irrep.

    Convenience wrapper around :func:`build_drt` that enforces spatial
    symmetry filtering. The spatial symmetry is tracked as the XOR
    product of irreps of singly occupied orbitals along the path.

    Parameters
    ----------
    norb : int
        Number of spatial orbitals.
    nelec : int
        Number of electrons.
    twos_target : int
        Target spin (2S).
    orbsym : sequence of int
        Bit-coded spatial irrep for each orbital.
    wfnsym : int
        Target wavefunction spatial irrep.

    Returns
    -------
    DRT
        Symmetry-filtered Distinct Row Table.
    """

    return build_drt(
        int(norb),
        int(nelec),
        int(twos_target),
        orbsym=orbsym,
        wfnsym=int(wfnsym),
    )
