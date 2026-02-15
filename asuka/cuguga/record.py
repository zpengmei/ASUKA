from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any

from asuka.cuguga.drt import STEP_ORDER, build_drt


def count_csfs(nspin: int, twos: int) -> int:
    """Count CSFs for *nspin* unpaired electrons with total spin quantum number *twos*.

    Uses the Paldus genealogical coupling formula via a dynamic programming
    table over spin-coupling paths.

    Parameters
    ----------
    nspin : int
        Number of singly occupied (unpaired) orbitals.
    twos : int
        Twice the total spin quantum number (2S).

    Returns
    -------
    int
        Number of CSFs.
    """
    if nspin < 0:
        raise ValueError("nspin must be >= 0")
    if twos < 0:
        raise ValueError("twos must be >= 0")
    if (nspin - twos) % 2 != 0:
        raise ValueError("nspin and twos must have the same parity")
    if twos > nspin:
        return 0

    n0 = (nspin - twos) // 2
    n1 = (nspin + twos) // 2

    gentable = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    for i1 in range(n0, n1 + 1):
        gentable[n0][i1] = 1
    for i0 in range(n0, 0, -1):
        row = gentable[i0][i0:]
        csum = 0
        newrow: list[int] = []
        for x in reversed(row):
            csum += x
            newrow.append(csum)
        newrow.reverse()
        newrow = [newrow[0]] + newrow
        gentable[i0 - 1][i0 - 1 :] = newrow
    return int(gentable[0][0])


def reference_count_all_csfs(norb: int, nelec: int, twos: int) -> int:
    """Reference CSF count by summing over all pairing schemes.

    This provides an independent cross-check for the DRT walk count by
    enumerating over all possible numbers of doubly-occupied orbitals.

    Parameters
    ----------
    norb : int
        Number of spatial orbitals.
    nelec : int
        Number of electrons.
    twos : int
        Twice the total spin (2S).

    Returns
    -------
    int
        Total number of CSFs.
    """
    norb = int(norb)
    nelec = int(nelec)
    twos = int(twos)
    if norb < 0:
        raise ValueError("norb must be >= 0")
    if nelec < 0:
        raise ValueError("nelec must be >= 0")
    if nelec > 2 * norb:
        raise ValueError("nelec must be <= 2*norb")
    if twos < 0:
        raise ValueError("twos must be >= 0")
    if twos > nelec:
        raise ValueError("twos must be <= nelec")
    if (nelec - twos) % 2 != 0:
        raise ValueError("nelec and twos must have the same parity")

    neleca = (nelec + twos) // 2
    nelecb = nelec - neleca
    min_npair = max(0, nelec - norb)
    max_npair = min(neleca, nelecb)

    total = 0
    for npair in range(min_npair, max_npair + 1):
        nspin = nelec - 2 * npair
        nfreeorb = norb - npair
        total += (
            math.comb(norb, npair)
            * math.comb(nfreeorb, nspin)
            * count_csfs(nspin, twos)
        )
    return int(total)


def case_snapshot(
    norb: int,
    nelec: int,
    twos: int,
    *,
    max_paths: int = 12,
    include_nodes: bool | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable snapshot of a DRT test case.

    Parameters
    ----------
    norb, nelec, twos : int
        Active space parameters.
    max_paths : int, optional
        Maximum number of sample CSF paths to include.
    include_nodes : bool or None, optional
        Whether to include the full node table. Defaults to *True* when the
        DRT has 8 or fewer nodes.

    Returns
    -------
    dict
        Nested dict with keys ``"case"``, ``"drt"``, and ``"reference"``.
    """
    drt = build_drt(norb=norb, nelec=nelec, twos_target=twos)
    if include_nodes is None:
        include_nodes = int(drt.nnodes) <= 8

    nodes_sorted: list[dict[str, Any]] = []
    for node_id in drt.iter_nodes_sorted():
        child = {}
        for sidx, sname in enumerate(STEP_ORDER):
            cid = int(drt.child[node_id, sidx])
            if cid < 0:
                child[sname] = None
            else:
                child[sname] = list(drt.node_state(cid))
        nodes_sorted.append(
            {
                "k": int(drt.node_k[node_id]),
                "ne": int(drt.node_ne[node_id]),
                "twos": int(drt.node_twos[node_id]),
                "nwalks": int(drt.nwalks[node_id]),
                "child": child,
            }
        )
    nodes_digest = hashlib.sha256(
        json.dumps(nodes_sorted, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    sample_paths = []
    for i in range(min(int(drt.ncsf), int(max_paths))):
        sample_paths.append(drt.path_as_str(drt.index_to_path(i)))

    drt_part: dict[str, Any] = {
        "ncsf": int(drt.ncsf),
        "nnodes": int(drt.nnodes),
        "leaf_state": list(drt.node_state(drt.leaf)),
        "sample_paths": sample_paths,
        "nodes_sha256": nodes_digest,
    }
    if include_nodes:
        drt_part["nodes_sorted"] = nodes_sorted

    return {
        "case": {"norb": int(norb), "nelec": int(nelec), "twos": int(twos)},
        "drt": drt_part,
        "reference": {"ncsf_csf_fci_style": reference_count_all_csfs(norb, nelec, twos)},
    }


def snapshot_to_json(snapshot: dict[str, Any]) -> str:
    """Serialize a snapshot dict to a pretty-printed JSON string."""
    return json.dumps(snapshot, indent=2, sort_keys=True) + "\n"


def write_snapshot(path: str | Path, snapshot: dict[str, Any]) -> None:
    """Write a snapshot dict to *path* as JSON."""
    path = Path(path)
    path.write_text(snapshot_to_json(snapshot))


def build_snapshot(cases: list[tuple[int, int, int]]) -> dict[str, Any]:
    """Build a combined snapshot for multiple ``(norb, nelec, twos)`` cases."""
    return {"cases": [case_snapshot(*c) for c in cases]}
