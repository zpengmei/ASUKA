from __future__ import annotations

"""Finite-difference nuclear gradients for ASUKA-native MRCISD.

This backend is intended for debugging/validation on small systems. It is
expensive because it rebuilds SCF → CASSCF → MRCISD at every displaced geometry.
"""

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from asuka.mrci.common import assign_roots_by_overlap
from asuka.mrci.driver_asuka import mrci_states_from_ref
from asuka.mrci.result import MRCIStatesResult


@dataclass(frozen=True)
class MRCIFDPoint:
    coords_bohr: np.ndarray
    e_tot: np.ndarray  # (nstate,)


def _infer_hf_method(scf_out: Any) -> str:
    """Map `SCFResult.method` to frontend `run_hf_df(method=...)`."""

    scf = getattr(scf_out, "scf", None)
    method = "" if scf is None else str(getattr(scf, "method", "")).strip().upper()
    if method.startswith("RHF"):
        return "rhf"
    if method.startswith("ROHF"):
        return "rohf"
    if method.startswith("UHF"):
        return "uhf"
    # Best-effort default.
    return "rhf"


def _infer_backend_from_scf_out(scf_out: Any) -> Literal["cpu", "cuda"]:
    B = getattr(scf_out, "df_B", None)
    try:
        import cupy as cp  # noqa: PLC0415

        if isinstance(B, cp.ndarray):  # type: ignore[attr-defined]
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _mol_with_coords_like(mol0: Any, coords_bohr: np.ndarray):
    from asuka.frontend.molecule import Molecule  # noqa: PLC0415

    coords = np.asarray(coords_bohr, dtype=np.float64).reshape((-1, 3))
    natm = int(coords.shape[0])
    atoms = [(str(mol0.atom_symbol(i)), coords[i].copy()) for i in range(natm)]
    return Molecule.from_atoms(
        atoms,
        unit="Bohr",
        charge=int(getattr(mol0, "charge", 0)),
        spin=int(getattr(mol0, "spin", 0)),
        basis=getattr(mol0, "basis", None),
        cart=bool(getattr(mol0, "cart", True)),
    )


def mrci_grad_states_from_ref_fd(
    scf_out0: Any,
    ref0: Any,
    *,
    mrci_states: MRCIStatesResult,
    roots: np.ndarray,
    states: Sequence[int],
    fd_step_bohr: float = 1e-3,
    which: Sequence[tuple[int, int]] | None = None,
    max_virt_e: int = 2,
    root_follow: Literal["hungarian", "greedy"] = "hungarian",
) -> list[np.ndarray]:
    """Finite-difference gradients for MRCISD total energies (Eh/Bohr).

    Notes
    -----
    This implementation performs a *cold-start* rebuild of SCF, CASSCF, and MRCISD
    at each displaced geometry. It uses overlap-based root assignment inside each
    MRCISD run to associate roots with the requested reference states.
    """

    from asuka.frontend.scf import run_hf_df  # noqa: PLC0415
    from asuka.mcscf.casscf import run_casscf  # noqa: PLC0415

    states_list = [int(s) for s in states]
    roots = np.asarray(roots, dtype=np.int64).ravel()
    if roots.shape != (len(states_list),):
        raise ValueError("roots must have shape (len(states),)")

    delta = float(fd_step_bohr)
    if delta <= 0.0:
        raise ValueError("fd_step_bohr must be > 0")

    mol0 = getattr(ref0, "mol", None)
    if mol0 is None:
        mol0 = getattr(scf_out0, "mol", None)
    if mol0 is None:
        raise ValueError("ref0.mol or scf_out0.mol is required")

    coords0 = np.asarray(getattr(mol0, "coords_bohr"), dtype=np.float64).reshape((-1, 3))
    natm = int(coords0.shape[0])
    if natm == 0:
        return [np.zeros((0, 3), dtype=np.float64) for _ in states_list]

    hf_method = _infer_hf_method(scf_out0)
    backend = _infer_backend_from_scf_out(scf_out0)
    auxbasis = getattr(scf_out0, "auxbasis_name", "autoaux")

    ncore = int(getattr(ref0, "ncore", 0))
    ncas = int(getattr(ref0, "ncas", 0))
    nelecas = getattr(ref0, "nelecas")
    nroots_ref = int(getattr(ref0, "nroots", 1))
    root_weights = getattr(ref0, "root_weights", None)

    # Energy evaluator returning energies for the requested state list.
    def _energies_at(coords_bohr: np.ndarray) -> np.ndarray:
        mol = _mol_with_coords_like(mol0, coords_bohr)
        scf_out = run_hf_df(
            mol,
            method=str(hf_method),
            backend=str(backend),
            df=True,
            auxbasis=auxbasis,
            guess=scf_out0,
        )
        casscf = run_casscf(
            scf_out,
            ncore=int(ncore),
            ncas=int(ncas),
            nelecas=nelecas,
            backend=str(backend),
            df=True,
            guess=ref0,
            nroots=int(nroots_ref),
            root_weights=None if root_weights is None else np.asarray(root_weights, dtype=np.float64).ravel().tolist(),
        )
        mrci_disp = mrci_states_from_ref(
            casscf,
            scf_out=scf_out,
            method="mrcisd",
            states=states_list,
            max_virt_e=int(max_virt_e),
        )
        ov = np.asarray(mrci_disp.mrci.overlap_ref_root, dtype=np.float64)
        roots_disp = assign_roots_by_overlap(ov, method=str(root_follow))
        e = np.asarray(mrci_disp.mrci.e_mrci, dtype=np.float64).ravel()
        return np.asarray([float(e[int(roots_disp[i])]) for i in range(len(states_list))], dtype=np.float64)

    # Central-difference loop (optionally restricted to a subset of coordinates).
    e0 = _energies_at(coords0.copy())
    grads = [np.zeros((natm, 3), dtype=np.float64) for _ in states_list]

    if which is None:
        which_list = [(ia, ax) for ia in range(natm) for ax in range(3)]
    else:
        which_list = [(int(ia), int(ax)) for ia, ax in which]
        if not which_list:
            raise ValueError("which must be non-empty when provided")
        if len(set(which_list)) != len(which_list):
            raise ValueError("which must not contain duplicates")
        for ia, ax in which_list:
            if ia < 0 or ia >= natm:
                raise ValueError(f"atom index out of range: {ia}")
            if ax < 0 or ax >= 3:
                raise ValueError(f"axis index out of range: {ax}")

    for ia, ax in which_list:
            coords_p = coords0.copy()
            coords_m = coords0.copy()
            coords_p[ia, ax] += delta
            coords_m[ia, ax] -= delta
            e_p = _energies_at(coords_p)
            e_m = _energies_at(coords_m)
            de = (e_p - e_m) / (2.0 * delta)
            for i in range(len(states_list)):
                grads[i][ia, ax] = float(de[i])

    return grads


__all__ = ["MRCIFDPoint", "mrci_grad_states_from_ref_fd"]
