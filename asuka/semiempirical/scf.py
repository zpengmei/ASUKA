"""RHF SCF driver for NDDO semiempirical methods.

Solves the orthogonal Roothaan equations with DIIS convergence acceleration.
In NDDO, the secular equation is a standard eigenproblem (no overlap
in the Roothaan equation) because AOs on different atoms are orthogonal
by the ZDO approximation.

Reference: MOPAC iter.F90
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .overlap import build_overlap_matrix
from .params import MethodParams

_NDDO_CORE_REQUIRED_SYMBOLS = (
    "build_ao_offsets",
    "build_core_hamiltonian",
    "build_core_hamiltonian_from_pair_terms",
    "build_fock",
    "build_onecenter_eris",
    "build_pair_ri_payload",
    "build_pair_list",
    "build_two_center_integrals",
    "compute_all_multipole_params",
    "core_core_repulsion",
    "core_core_repulsion_from_gamma_ss",
    "nao_for_Z",
    "valence_electrons",
)
_NDDO_CORE_API: Dict[str, object] | None = None
_NDDO_CORE_ERR: Exception | None = None


def _validate_fock_mode(fock_mode: str) -> str:
    mode = str(fock_mode).strip().lower()
    if mode not in ("ri", "w", "auto"):
        raise ValueError("fock_mode must be 'ri', 'w', or 'auto'")
    return mode


def _require_nddo_core_api() -> Dict[str, object]:
    """Import and validate required asuka.nddo_core symbols once."""
    global _NDDO_CORE_API, _NDDO_CORE_ERR
    if _NDDO_CORE_API is not None:
        return _NDDO_CORE_API
    if _NDDO_CORE_ERR is not None:
        raise RuntimeError(
            "AM1 requires asuka.nddo_core, but previous contract validation failed"
        ) from _NDDO_CORE_ERR

    try:
        import asuka.nddo_core as nddo_core  # type: ignore
    except Exception as exc:
        _NDDO_CORE_ERR = exc
        raise RuntimeError(
            "AM1 requires asuka.nddo_core. Ensure the package is installed from a tree "
            "that includes asuka/nddo_core and retry."
        ) from exc

    missing = [name for name in _NDDO_CORE_REQUIRED_SYMBOLS if not hasattr(nddo_core, name)]
    if missing:
        err = RuntimeError(
            "asuka.nddo_core is missing required AM1 symbols: " + ", ".join(sorted(missing))
        )
        _NDDO_CORE_ERR = err
        raise err

    _NDDO_CORE_API = {name: getattr(nddo_core, name) for name in _NDDO_CORE_REQUIRED_SYMBOLS}
    return _NDDO_CORE_API


@dataclass
class SCFResult:
    """Result of an NDDO SCF calculation."""
    converged: bool
    n_iter: int
    energy_electronic: float  # Hartree
    energy_core: float        # Hartree
    energy_total: float       # Hartree
    eps: np.ndarray           # Orbital energies
    C: np.ndarray             # MO coefficients
    P: np.ndarray             # Density matrix
    F: np.ndarray             # Final Fock matrix
    H: np.ndarray             # Core Hamiltonian
    nocc: int


def _solve_roothaan(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve orthogonal Roothaan equation FC = CE."""
    eps, C = np.linalg.eigh(F)
    return eps, C


def _build_density(C: np.ndarray, nocc: int) -> np.ndarray:
    """Build closed-shell density matrix P = 2 * C_occ @ C_occ.T."""
    Cocc = C[:, :nocc]
    return 2.0 * (Cocc @ Cocc.T)


def am1_scf(
    atomic_numbers: Sequence[int],
    coords_bohr: np.ndarray,
    params: MethodParams,
    charge: int = 0,
    max_iter: int = 200,
    conv_tol: float = 1e-8,
    comm_tol: float = 1e-6,
    diis: bool = True,
    diis_start: int = 2,
    diis_size: int = 8,
    device: str = "cpu",
    fock_mode: str = "ri",
) -> SCFResult:
    """Run an AM1 SCF calculation.

    Parameters
    ----------
    atomic_numbers : sequence of int
        Atomic numbers.
    coords_bohr : (N, 3) array
        Coordinates in Bohr.
    params : MethodParams
        AM1 parameter set.
    charge : int
        Molecular charge.
    max_iter : int
        Maximum SCF iterations.
    conv_tol : float
        Energy convergence threshold (Hartree).
    comm_tol : float
        Commutator convergence threshold on ``max(abs(FP-PF))``.
    diis : bool
        Use DIIS convergence acceleration.
    diis_start : int
        Start DIIS after this many iterations.
    diis_size : int
        Maximum number of DIIS vectors.
    device : str
        Execution device ("cpu" or "cuda").
    fock_mode : str
        Two-center Fock mode (``"ri"``, ``"w"``, or ``"auto"``). ``"auto"``
        is resolved by the CUDA path; the CPU path ignores this knob.

    Returns
    -------
    SCFResult
    """
    device = str(device).strip().lower()
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    fock_mode = _validate_fock_mode(fock_mode)

    nddo = _require_nddo_core_api()
    build_ao_offsets = nddo["build_ao_offsets"]
    build_core_hamiltonian = nddo["build_core_hamiltonian"]
    build_fock = nddo["build_fock"]
    build_onecenter_eris = nddo["build_onecenter_eris"]
    build_pair_list = nddo["build_pair_list"]
    build_two_center_integrals = nddo["build_two_center_integrals"]
    compute_all_multipole_params = nddo["compute_all_multipole_params"]
    core_core_repulsion = nddo["core_core_repulsion"]
    nao_for_Z = nddo["nao_for_Z"]
    valence_electrons = nddo["valence_electrons"]

    if device == "cuda":
        from .gpu.scf_gpu import am1_scf_cuda

        return am1_scf_cuda(
            atomic_numbers=atomic_numbers,
            coords_bohr=coords_bohr,
            params=params,
            charge=charge,
            max_iter=max_iter,
            conv_tol=conv_tol,
            comm_tol=comm_tol,
            diis=diis,
            diis_start=diis_start,
            diis_size=diis_size,
            fock_mode=fock_mode,
        )

    atomic_numbers = list(atomic_numbers)
    coords_bohr = np.asarray(coords_bohr, dtype=float)
    N = len(atomic_numbers)
    elem_params = params.elements

    # Validate elements
    for Z in atomic_numbers:
        if Z not in elem_params:
            raise ValueError(f"No parameters for element Z={Z}")

    # AO bookkeeping
    offsets = build_ao_offsets(atomic_numbers)
    nao_total = offsets[-1]

    # Number of electrons
    n_elec = sum(valence_electrons(Z) for Z in atomic_numbers) - charge
    if n_elec % 2 != 0:
        raise ValueError(f"Odd number of electrons ({n_elec}), open-shell not supported")
    nocc = n_elec // 2

    # Pair list
    pair_i, pair_j, _, pair_r = build_pair_list(coords_bohr)

    # Multipole parameters
    mp_params = compute_all_multipole_params(elem_params)

    # STO overlap matrix
    S = build_overlap_matrix(atomic_numbers, coords_bohr, elem_params)

    # Two-center integrals
    W_list = build_two_center_integrals(
        atomic_numbers, coords_bohr, pair_i, pair_j, mp_params
    )

    # Core Hamiltonian
    H = build_core_hamiltonian(
        atomic_numbers, coords_bohr, S, pair_i, pair_j, W_list, elem_params
    )

    # Core-core repulsion
    E_core = core_core_repulsion(
        atomic_numbers, coords_bohr, pair_i, pair_j, pair_r, W_list, elem_params
    )

    # One-center ERIs per atom
    onecenter_eris = []
    for Z in atomic_numbers:
        ep = elem_params[Z]
        nao = nao_for_Z(Z)
        G = build_onecenter_eris(nao, ep.gss, ep.gsp, ep.gpp, ep.gp2, ep.hsp)
        onecenter_eris.append(G)

    # Initial guess: solve FC = CE with F = H
    eps, C = _solve_roothaan(H)
    P = _build_density(C, nocc)

    # SCF loop
    E_old = 0.0
    converged = False
    diis_F_list: List[np.ndarray] = []
    diis_err_list: List[np.ndarray] = []

    for it in range(max_iter):
        # Build Fock matrix
        F = build_fock(
            H, P, atomic_numbers, offsets, onecenter_eris,
            pair_i, pair_j, W_list,
        )
        err = F @ P - P @ F
        err_max = float(np.max(np.abs(err)))

        # Electronic energy: E_el = 0.5 * Tr(P * (H + F))
        E_el = 0.5 * np.sum(P * (H + F))
        E_total = E_el + E_core

        # Check convergence
        dE = abs(E_total - E_old)
        if it > 0 and dE < conv_tol and err_max < comm_tol:
            converged = True
            break
        E_old = E_total

        # DIIS acceleration
        if diis and it >= diis_start:
            diis_F_list.append(F.copy())
            diis_err_list.append(err.ravel())

            if len(diis_F_list) > diis_size:
                diis_F_list.pop(0)
                diis_err_list.pop(0)

            if len(diis_F_list) >= 2:
                m = len(diis_F_list)
                Bmat = np.empty((m + 1, m + 1), dtype=float)
                for i in range(m):
                    for j in range(m):
                        Bmat[i, j] = np.dot(diis_err_list[i], diis_err_list[j])
                Bmat[:m, m] = -1.0
                Bmat[m, :m] = -1.0
                Bmat[m, m] = 0.0
                rhs = np.zeros(m + 1, dtype=float)
                rhs[m] = -1.0
                try:
                    coeff = np.linalg.solve(Bmat, rhs)[:m]
                    F = sum(c * Fi for c, Fi in zip(coeff, diis_F_list))
                except np.linalg.LinAlgError:
                    pass

        # Diagonalize
        eps, C = _solve_roothaan(F)
        P = _build_density(C, nocc)

    return SCFResult(
        converged=converged,
        n_iter=it + 1,
        energy_electronic=E_el,
        energy_core=E_core,
        energy_total=E_total,
        eps=eps,
        C=C,
        P=P,
        F=F,
        H=H,
        nocc=nocc,
    )
